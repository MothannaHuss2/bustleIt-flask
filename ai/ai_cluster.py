import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import datetime
from collections import defaultdict, OrderedDict
from pydantic import BaseModel
from typing import List, Dict, Any
import utils.api as api
import random
from customTypes import ClusteredProfile, BatchedTasks, Schedule, DailySchedule, DailyTask, RawProfile
from sentence_transformers import SentenceTransformer
from logger import get_logger
import time

logger = get_logger('AI')
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
class AiClusteredProfile(BaseModel):
    id: str
    introverted: float
    extraverted: float
    observant: float
    intuitive: float
    thinking: float
    feeling: float
    judging: float
    prospecting: float
    assertive: float
    turbulent: float
    preferences: List[str]
    cluster: int

class data(BaseModel):
    user_id: str
    scores: Dict[str, float]
    preferences: List[str]
    
# # model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
def cluster_single_record(profile: RawProfile) -> int:
    """
    Clusters a single record based on personality traits.

    Parameters:
    profile (RawProfile): A RawProfile object containing the user's data.
    kmeans (KMeans): Trained KMeans model for clustering.
    scaler (MinMaxScaler): Trained MinMaxScaler for scaling personality traits.

    Returns:
    int: The cluster assignment for the single data point.
    """
    # Prepare personality traits data only (exclude preferences if KMeans was trained on traits only)
    processed = process(profile)
    df = pd.DataFrame([processed])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    traits = ["Extraversion", "Intuition", "Thinking", "Judging", "Assertiveness"]
    df[traits] = scaler.fit_transform(df[traits])
    mlb = MultiLabelBinarizer()
    preferences_encoded = mlb.fit_transform(df["preferences"])
    preferences_df = pd.DataFrame(preferences_encoded, columns=mlb.classes_)  # type: ignore
    df = pd.concat(
        [df.reset_index(drop=True), preferences_df.reset_index(drop=True)],
        axis=1,
    )
    traits_df = df[traits]
    model = joblib.load("models/model.pkl")
    values = traits_df.values
    cluster = model.predict(values)[0]

    return cluster

def process(profile:data):
    """
    This function processes a new user's raw input into a feature vector by computing differences
    between personality trait scores, and scales the resulting vector to a range of [-1, 1].

    Input:
        - user (list): A list representing the raw personality trait scores of the user.

    Output:
        - Returns a normalized feature vector (list) for the user after processing and scaling.
    """
    new_vector: list[float] = []

    # Calculate the differences between paired personality traits.
    extraversion = profile.scores['extraverted'] - profile.scores['introverted']
    intuition = profile.scores['intuitive'] - profile.scores['observant']
    thinking = profile.scores['thinking'] - profile.scores['feeling']
    judging = profile.scores['judging'] - profile.scores['prospecting']
    assertiveness = profile.scores['assertive'] - profile.scores['turbulent']
    return {
        "Extraversion": extraversion,
        "Intuition": intuition,
        "Thinking": thinking,
        "Judging": judging,
        "Assertiveness": assertiveness,
        'preferences': profile.preferences
        
    }

def train(k) -> List[int]:
    try:
        clustered_users= api.getAllUsers()
        processed = [
            process(user) for user in clustered_users
        ]
        
        ids = [user.user_id for user in clustered_users]
        
        
        df = pd.DataFrame(processed)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        traits = ["Extraversion", "Intuition", "Thinking", "Judging", "Assertiveness"]
        df[traits] = scaler.fit_transform(df[traits])
        mlb = MultiLabelBinarizer()
        preferences_encoded = mlb.fit_transform(df["preferences"])
        preferences_df = pd.DataFrame(preferences_encoded, columns=mlb.classes_)  # type: ignore

        df = pd.concat(
                [df.reset_index(drop=True), preferences_df.reset_index(drop=True)],
                axis=1,
            )
        
        X = df[traits].values
        
        kmeans = KMeans(n_clusters=k, random_state=42)    
        clusters = kmeans.fit_predict(X)
        df['cluster'] = clusters
        json_clusters = []
        for i in range(len(df)):
            integer_cluster_value = int(df["cluster"].iloc[i])
            json_clusters.append({"id": ids[i], "cluster": integer_cluster_value})
        
        
        joblib.dump(kmeans, "models/model_1.pkl")
        return json_clusters
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def getScores(user: ClusteredProfile) -> list[float]:
    scores = user.scores
    new_user = [
            scores['introverted'], 
            scores['extraverted'], 
            scores['observant'], 
            scores['intuitive'], 
            scores['thinking'], 
            scores['feeling'], 
            scores['judging'], 
            scores['prospecting'], 
            scores['assertive'], 
            scores['turbulent']
                    
        ]
    return new_user
def getTasks(id,range, startDate,prefs, dict=True):
    
   try:
        tasks = api.getRangedSchedule(id, startDate, range)
        if not tasks:
            raise Exception("No tasks found")
        data = tasks.data
        schedules = list(data.values())
        tasks = [
            schedule.tasks for schedule in schedules
        ]
        tasks_flat = [
                {
                'id':task.task_id,
                'name': task.name,
                'category': task.category,
                'startTime': task.start_time,
                'endTime': task.end_time,
                'completed': task.completed,
                "created_at": datetime.datetime.strptime(task.created_at.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S.%f"),
                "updated_at": datetime.datetime.strptime(task.updated_at.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S.%f")
                
            } 
            for sublist in 
            tasks for 
            task in sublist
            if task.category in prefs
            ]

        
        return tasks_flat
   except Exception as e:
        print(f"An error occurred in getTasks: {e}")
        return []


def calculate_weighted_engagement(user_tasks: list[DailyTask], max_minutes):
    from datetime import datetime

    try:
        user_history_engagement = 0
        total_weight = 0
        current_time = datetime.now().time()

        for task in user_tasks:
           
            task_time = datetime.strptime(task.start_time, '%H:%M').time()  # Parse time-only string

            # Calculate difference in minutes
            time_diff = abs((datetime.combine(datetime.today(), current_time) - 
                            datetime.combine(datetime.today(), task_time)).total_seconds() / 60)

            if 0 <= time_diff <= max_minutes:
                weight = 1.0 - (time_diff / max_minutes)  # More recent tasks have higher weight
                completion_rate = 1 if task.completed else 0
                user_history_engagement += completion_rate * weight
                total_weight += weight

        weighted_engagement_rate = user_history_engagement / total_weight if total_weight else 0
        return weighted_engagement_rate
    except Exception as e:
        print(f"An error occurred in calculate_weighted_engagement: {e}")
        return 0


def turnToDict(tasks):
    
    logger.info(f"\nTasks: {tasks[-1]}\n")
    try:
        return [
        {
                'id':task.task_id,
                'name': task.name,
                'category': task.category,
                'startTime': task.start_time,
                'endTime': task.end_time,
                'completed': task.completed,
                "created_at": task.created_at,
                "updated_at": task.updated_at
                
            } 
        for task in tasks
    ]
    except Exception as e:
        print(f"An error occurred in turnToDict: {e}")
        return []
def recommend(
user: ClusteredProfile, 
range=30,
work_end_time=17,
sleep_time=24, 
work_start_time=8, 
with_time=True, 
day=None,
givenTasks = None,
userGivens=None
):
    """_summary_

    Args:
        user (api.ClusteredProfile): _description_
        range (_type_): _description_
        work_end_time (int, optional): _description_. Defaults to 17.
        sleep_time (int, optional): _description_. Defaults to 24.
        work_start_time (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    try:
        logger.info(f"\nGivens: {givenTasks[0]}\n")
        logger.info(f"\nUser givens: {userGivens[0]}\n")
        if givenTasks is not None:
            givenTasks = turnToDict(givenTasks)
        user_id = user.user_id
        cluster = user.cluster
        ids = []
        preferences = np.array(user.preferences) 
        if givenTasks is None:
            scores = np.array(user.scores)  
            new_user = np.array(getScores(user)) 
            
            people_in_cluster = api.getUsersByCluster(cluster)
            similar_ids = np.array([user.user_id for user in people_in_cluster]) 
            
            scores = np.array([getScores(user) for user in people_in_cluster])  
            
            reshaped = np.reshape(new_user, (1, -1))  
            similarity_scores = cosine_similarity(reshaped, scores)  
            top_similar_indices = np.argsort(similarity_scores[0])[-5:]  
            ids = list(similar_ids[top_similar_indices])  
            ids.extend(['542172eb-c417-46c0-b9b1-78d1b7630bf5', user_id])  
        
        # tasks = [
        #     getTasks(id, range, datetime.datetime.now().strftime('%Y-%m-%d'), preferences) for id in ids
        # ] if tasks is None else givenTasks
        user_tasks_og = getTasks(user_id, range, datetime.datetime.now().strftime('%Y-%m-%d'), preferences, True)
        
        others = api.getBatchedTasks(ids) if givenTasks is None else givenTasks
        user_tasks = [task for task in others if task.user_id == user_id] if givenTasks is None else userGivens
        
        all_tasks = [
            np.array(task.all_tasks) for task in others  
        ] if givenTasks is None else givenTasks
        logger.info(f"\nUser tasks: {all_tasks[0]}\n")
        flat = np.array([task for sublist in all_tasks for task in sublist]) if givenTasks is None else  np.array(givenTasks) # Flattened tasks
        logger.info(f"\nFlattened {flat[0]}\n")
        
        engagement_rate = 0.6
        similar_users_engagement_rate = 1 - engagement_rate
        
        numberOfTasks = 5
        tasks_from_user_history = int(numberOfTasks * engagement_rate)
        tasks_from_others = int(numberOfTasks * similar_users_engagement_rate)
        
        user_sample = user_tasks[0].all_tasks if userGivens is None else userGivens
        
        sample_from_user_history = random.sample(user_sample, tasks_from_user_history)
        logger.info(f"\nSample from user history: {sample_from_user_history[0]}\n")
        sample_from_others = random.sample(flat.tolist(), tasks_from_others)
        logger.info(f"\nSample from others: {sample_from_others[0]}\n")
        
        today = getTodayName() if day is None else day
        concatenated = sample_from_user_history+ sample_from_others
        
        logger.info(f"\nConcatenated: {concatenated[-1]}\n")
        all_tasks_combined = [
            {
                'name': task['name'],
                'category': task['category'],
                'startTime': task['startTime'],
                'endTime': task['endTime'],
                'completed': task['completed'],
            }
            for task in concatenated
            if getDateDay(task['created_at']) == today
        ]
        
        x = recommend_task_start_times(
            all_tasks_combined,
            user_tasks_og,
            work_end_time,
            sleep_time,
            model=model
        )
        return x if with_time else all_tasks_combined
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}




def get_name_similarity(task1: Dict[str, Any], task2: Dict[str, Any], model) -> float:
    """
    Calculate the cosine similarity between the names of two tasks.

    Args:
        task1 (dict): First task with at least a 'name' key.
        task2 (dict): Second task with at least a 'name' key.
        model: Model with an 'encode' method to generate embeddings.

    Returns:
        float: Cosine similarity between the task names.
    """
    try:
        task1_name = task1['name']
        task2_name = task2['name']
    except KeyError as e:
        raise KeyError(f"Missing 'name' key in task: {e}")

    task1_embedding = model.encode(task1_name)
    task2_embedding = model.encode(task2_name)

    task1_embedding = np.atleast_2d(task1_embedding)
    task2_embedding = np.atleast_2d(task2_embedding)

    similarity = cosine_similarity(task1_embedding, task2_embedding)[0][0]

    return similarity

def calculate_category_frequency(existing_tasks: List[Dict[str, Any]], work_end_time: int, sleep_time: int) -> Dict[str, Dict[int, int]]:

    category_frequency = defaultdict(lambda: defaultdict(int))

    for task in existing_tasks:
        try:
            start_hour = int(task['startTime'].split(":")[0])
            if work_end_time <= start_hour < sleep_time:
                category_frequency[task['category']][start_hour] += 1
        except Exception as e:
            print("Error:", e)

    return category_frequency # type: ignore
def recommend_task_start_times(
    new_tasks: List[Dict[str, Any]],
    existing_tasks: List[Dict[str, Any]],
    work_end_time: int,
    sleep_time: int,
    model,
    n: int = 5,
    time_gap: int = 1,
    similarity_threshold: float = 0.5
) -> Dict[str, str]:
    """
    Recommend start times for new tasks based on category frequency in specific time slots.

    Args:
        new_tasks (list): List of new tasks.
        existing_tasks (list): List of existing tasks.
        work_end_time (int): Hour when work ends (e.g., 17 for 5 PM).
        sleep_time (int): Hour when sleep time starts (e.g., 23 for 11 PM).
        model: Model with an 'encode' method to generate embeddings.
        n (int): Number of top similar tasks to consider.
        time_gap (int): Required gap between tasks in hours for dissimilar tasks.
        similarity_threshold (float): Threshold above which tasks are considered similar.

    Returns:
        dict: Mapping from task names to scheduled start times in 'HH:MM' format.
    """
    try:
        category_frequency = calculate_category_frequency(existing_tasks, work_end_time, sleep_time)

        available_times = []
        for hour in range(work_end_time, sleep_time):
            available_times.append(datetime.datetime.strptime(f"{hour}:00", "%H:%M"))
        available_times.sort()

        scheduled_tasks = {}
        last_scheduled_time = None
        last_task_name = None

        for new_task in new_tasks:
            category = new_task.get('category', None)

            if category in category_frequency:
                peak_hour = max(category_frequency[category], key=category_frequency[category].get)
                recommended_time = datetime.datetime.strptime(f"{peak_hour}:00", "%H:%M")
            else:
                recommended_time = available_times[0] 

            if last_scheduled_time is None:
                earliest_start_time = available_times[0]
            else:
                last_task = {'name': last_task_name}
                similarity_with_last = get_name_similarity(new_task, last_task, model)

                if similarity_with_last >= similarity_threshold:
                    earliest_start_time = last_scheduled_time
                else:
                    earliest_start_time = last_scheduled_time + datetime.timedelta(hours=time_gap)

            potential_times = [t for t in available_times if t >= earliest_start_time]
            if not potential_times:
                break

            scheduled_time = min(
                potential_times,
                key=lambda t: abs((t - recommended_time).total_seconds())
            )

            scheduled_tasks[new_task['name']] = scheduled_time.strftime('%H:%M')

            task_duration = datetime.timedelta(hours=1)
            end_time = scheduled_time + task_duration
            available_times = [t for t in available_times if t < scheduled_time or t >= end_time]

            last_scheduled_time = end_time
            last_task_name = new_task['name']

        return scheduled_tasks
    except Exception as e:
        print(f"An error occurred in recommend_task_start_times: {e}")
        return {}

def get_next_saturday(from_date: datetime.datetime) -> datetime.datetime:
    """
    Get the next Saturday from a given date. If the given date is Saturday, return that date.
    """
    try:
        days_until_saturday = (5 - from_date.weekday()) % 7
        next_saturday = from_date + datetime.timedelta(days=days_until_saturday)
        return next_saturday
    except Exception as e:
        print(f"An error occurred in get_next_saturday: {e}")
        return from_date

def getTodayName():
    currentDay = datetime.datetime.today().strftime('%A')
    return currentDay

def getDateDay(date):
    try:

        if isinstance(date, str):

            if date.endswith("Z"):
                date = date[:-1]  
            parsed_date = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
        elif isinstance(date, datetime.datetime):

            parsed_date = date
        else:
            raise ValueError("Invalid date format. Input must be a string or datetime object.")
        

        day_name = parsed_date.strftime("%A")
        return day_name
    except Exception as e:
        print(f"An error occurred in getDateDay: {e}")
        return ""

def get_recommended_tasks(
    similar_users_tasks: List[Dict[str, Any]],
    user_tasks_df: pd.DataFrame,
    portion_from_user_tasks: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Generates a list of recommended tasks, including a portion from the user's own tasks.
    """
    try:
        if not similar_users_tasks:
            return []
            
        portion_from_user_tasks = max(0.0, min(1.0, portion_from_user_tasks))  
        num_user_tasks = int(len(similar_users_tasks) * (1 - portion_from_user_tasks))
        
        user_own_tasks = []
        if num_user_tasks > 0 and not user_tasks_df.empty:
            sample_size = min(num_user_tasks, len(user_tasks_df))
            user_own_tasks = user_tasks_df.sample(n=sample_size, replace=False).to_dict('records')

        combined_tasks = similar_users_tasks + user_own_tasks
        seen_tasks = {}
        for task in combined_tasks:
            if 'name' in task:
                seen_tasks[task['name']] = task
                
        return list(seen_tasks.values())
        
    except Exception as e:
        print(f"Error in get_recommended_tasks: {e}")
        return similar_users_tasks  # Fallback to original tasks if error occur



def getAllUsersTasks() -> list[DailyTask]:
    try:
        tasks = api.getAllUsersTasks()
        return tasks
    except Exception as e:
        print(f'Error at getAllUsersTasks: {e}')
        return []
def getAllTasks(id):
    try:
        if not id.__class__ == list:
            id = [id]
        else:
            id = id
        tasks = api.getBatchedTasks(id)
        if len(id) == 1:
            return tasks[0].all_tasks
        elif len(id) > 1:
            return [task.all_tasks for task in tasks][0]
        else:
            return []
    except Exception as e:
        print(f"An error occurred in getAllTasks: {e}")
        return []

def findKSimilarUsers(people: List[ClusteredProfile], user: ClusteredProfile, k: int) -> List[ClusteredProfile]:
    """
    Finds the k most similar users to a given user based on cosine similarity of their personality traits.

    Args:
        people (list): List of users.
        user (api.ClusteredProfile): User to compare against.
        k (int): Number of similar users to find.

    Returns:
        list: List of k most similar users.
    """
    try:
        score = getScores(user)
        people_in_cluster = people
        similar_ids = [user.user_id for user in people_in_cluster]
        
        scores = [
            getScores(user) for user in people_in_cluster
        ]
        
        reshaped = np.reshape(score, (1, -1))
        similarity_scores = cosine_similarity(reshaped, scores)
        top_similar_indices = similarity_scores.argsort()[0][-5:]
        ids = [similar_ids[i] for i in top_similar_indices]
        return [user for user in people if user.user_id in ids]
    except Exception as e:
        print(f"An error occurred in findKSimilarUsers : {e}")
        return []




def getAll() -> list[BatchedTasks]:
    try:
        tasks = api.getAllTasks()
        return tasks
    except Exception as e:
        print(f"An error occurred in getAllTasks: {e}")
        return []
def filter(tasks: list[DailyTask], day) -> List[DailyTask]:
    return [task for task in tasks if getDateDay(task.created_at) == day], day
def getTime(date:str):
    parsed = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    return parsed.hour

def recommend_weekly_tasks(
    user: ClusteredProfile,
    work_end_time=17,
    sleep_time=24,
    work_start_time=8,
    portion_from_user_tasks=0.6,
    range=30
) -> dict:
    start = time.time()
    weekly_schedule = OrderedDict()
    try:
            
        user_tasks = getAllTasks(user.user_id)
        today = getTodayName()
        hashPrefs = {
            category:1
            for category in user.preferences
        }
        allUserTasks = getAllUsersTasks()
        logger.info(f"\nAll user tasks: {allUserTasks[0]}\n")
        filteredAll = [
            task for task in allUserTasks if task['category'] if hashPrefs.get(task['category'])
        ]
        logger.info(f"\nfilteredAll: {filteredAll[0]}\n")
        similar_users = findKSimilarUsers(api.getUsersByCluster(user.cluster), user, 5)
        similar_users_ids = [user.user_id for user in similar_users]
        
        
        similar_users_tasks = getAllTasks(similar_users_ids)
        userTs = getTasks(user.user_id, range, datetime.datetime.now().strftime('%Y-%m-%d'), user.preferences)
        logger.info(f"Similar users: {similar_users_tasks[0]}")
        similar_users_tasks = [task for task in similar_users_tasks if task.category in user.preferences]
        combined = similar_users_tasks + user_tasks
        # logger.info(f'Today is {today}')
        
        
        all_tasks = [
            {
                'name': task.name,
                'category': task.category,
            }
            for task in combined
        ]
        
        user_tasks_df = pd.DataFrame(all_tasks, columns=['name', 'category'])
        

        today = datetime.datetime.today()
        next_saturday = get_next_saturday(today)

        days_order = [
            "Saturday", "Sunday", "Monday", 
            "Tuesday", "Wednesday", "Thursday", "Friday"
        ]
        filtered = [
            filter(combined, day) for day in days_order
        ]
        logger.info(f'User tasks: {len(user_tasks_df)}')

        dic = {
            day: tasks for tasks, day in filtered if len(tasks) > 0
        }
        logger.info(f"Looping through days: {dic.keys()}")
        # Generate schedule for each day
        for i, day_name in enumerate(days_order):
            current_date = next_saturday + datetime.timedelta(days=i)
            if dic.get(day_name):
                logger.info(f"\nTasks: {dic[day_name][0]}\n")
                tasks_to_recommend = recommend(
                    user,
                    range,
                    work_end_time,
                    sleep_time,
                    work_start_time,
                    with_time=False,
                    day=day_name,
                    givenTasks=dic[day_name],
                    userGivens=userTs
                )
                try:
                    recommended_tasks = get_recommended_tasks(
                        tasks_to_recommend,
                        user_tasks_df,
                        portion_from_user_tasks,
                    )

                    scheduled_tasks = schedule_tasks_for_day(
                        recommended_tasks,
                        day_name,
                        work_end_time,
                        sleep_time,
                        work_start_time
                    )

                    daily_schedule = {
                        "day": day_name,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "tasks": scheduled_tasks
                    }

                    weekly_schedule[day_name] = daily_schedule
                    
                except Exception:
                    weekly_schedule[day_name] = {
                        "day": day_name,
                        "date": current_date.strftime("%Y-%m-%d"),
                        "tasks": [{"name": "blocked", 
                                "startTime": f"{work_start_time:02}:00",
                                "endTime": f"{work_end_time:02}:00"}]
                    }
            else:
                weekly_schedule[day_name] = {
                    "day": day_name,
                    "date": current_date.strftime("%Y-%m-%d"),
                    "tasks": [{"name": "blocked", 
                              "startTime": f"{work_start_time:02}:00",
                              "endTime": f"{work_end_time:02}:00"}]
                }
                
        end = time.time()
        logger.info(f'Execution time for recommend_weekly_tasks: {end - start} seconds')
        return dict(weekly_schedule)

    except Exception as e:
        print(f"Error in recommend_weekly_tasks: {e}")
        return {}

def recommend_daily(
user: ClusteredProfile, 
range=30,
work_end_time=17,
sleep_time=24, 
work_start_time=8, 
with_time=True, 
day=None):
    """_summary_

    Args:
        user (api.ClusteredProfile): _description_
        range (_type_): _description_
        work_end_time (int, optional): _description_. Defaults to 17.
        sleep_time (int, optional): _description_. Defaults to 24.
        work_start_time (int, optional): _description_. Defaults to 8.

    Returns:
        _type_: _description_
    """
    try:
        scores = np.array(user.scores)  
        preferences = np.array(user.preferences) 
        user_id = user.user_id
        cluster = user.cluster
        new_user = np.array(getScores(user)) 
        
        people_in_cluster = api.getUsersByCluster(cluster)
        similar_ids = np.array([user.user_id for user in people_in_cluster]) 
        
        scores = np.array([getScores(user) for user in people_in_cluster])  
        
        reshaped = np.reshape(new_user, (1, -1))  
        similarity_scores = cosine_similarity(reshaped, scores)  
        top_similar_indices = np.argsort(similarity_scores[0])[-5:]  
        ids = list(similar_ids[top_similar_indices])  
        ids.extend(['542172eb-c417-46c0-b9b1-78d1b7630bf5', user_id])  
        
        tasks = [
            getTasks(id, range, datetime.datetime.now().strftime('%Y-%m-%d'), preferences) for id in ids
        ]
        user_tasks_og = tasks[-1]
        
        others = api.getBatchedTasks(ids)
        user_tasks = [task for task in others if task.user_id == user_id]
        
        all_tasks = [
            np.array(task.all_tasks) for task in others  
        ]
        flat = np.array([task for sublist in all_tasks for task in sublist])  # Flattened tasks
        
        engagement_rate = 0.6
        similar_users_engagement_rate = 1 - engagement_rate
        
        numberOfTasks = 5
        tasks_from_user_history = int(numberOfTasks * engagement_rate)
        tasks_from_others = int(numberOfTasks * similar_users_engagement_rate)
        
        sample_from_user_history = np.array(random.sample(user_tasks[0].all_tasks, tasks_from_user_history))
        sample_from_others = np.array(random.sample(flat.tolist(), tasks_from_others))  
        
        today = getTodayName() if day is None else day
        all_tasks_combined = [
            {
                'name': task.name,
                'category': task.category,
                'startTime': task.start_time,
                'endTime': task.end_time,
                'completed': task.completed,
            }
            for task in np.concatenate((sample_from_user_history, sample_from_others))
            if getDateDay(task.created_at) == today
        ]
        
        x = recommend_task_start_times(
            all_tasks_combined,
            user_tasks_og,
            work_end_time,
            sleep_time,
            model=model
        )
        return x if with_time else all_tasks_combined
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

def schedule_tasks_for_day(
    tasks: List[Dict[str, Any]],
    day_name: str,
    work_end_time: int,
    sleep_time: int,
    work_start_time: int
) -> List[Dict[str, Any]]:
    """
    Schedules tasks for a specific day.

    Args:
        tasks (list): List of tasks to schedule.
        day_name (str): Name of the day.
        work_end_time (int): End of work day.
        sleep_time (int): Time the user goes to sleep.
        work_start_time (int): Start of work day.

    Returns:
        list: Scheduled tasks with start and end times.
    """
    scheduled_times = recommend_task_start_times(
        tasks,
        existing_tasks=[], 
        work_end_time=work_end_time,
        sleep_time=sleep_time,
        model=model
    )
    scheduled_tasks = [
        {
            "name": task_name,
            "startTime": start_time,
            "endTime": f"{(int(start_time[:2]) + 1):02}:00"
        }
        for task_name, start_time in scheduled_times.items()
    ]

    scheduled_tasks.insert(0, {
        "name": "blocked",
        "startTime": f"{work_start_time:02}:00",
        "endTime": f"{work_end_time:02}:00"
    })

    return scheduled_tasks