# # ============================ #
# # Logic serving the AI cluster #
# # ============================ #

# import json
# import random
# from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import datetime
from collections import defaultdict
from pydantic import BaseModel
from typing import List


class RawProfile(BaseModel):
    id: int
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
    profile_data = {
        "Extraversion": profile.extraverted - profile.introverted,
        "Intuition": profile.intuitive - profile.observant,
        "Thinking": profile.thinking - profile.feeling,
        "Judging": profile.judging - profile.prospecting,
        "Assertiveness": profile.assertive - profile.turbulent,
    }

    # Scale the personality traits
    traits_df = pd.DataFrame([profile_data])

    model = joblib.load("models/model.pkl")
    scaler = MinMaxScaler(feature_range=(-1, 1))

    traits_scaled = scaler.fit_transform(traits_df)

    # Predict the cluster for this single record (only using the traits)
    cluster = model.predict(traits_scaled)[0]

    return cluster

# def cluster(k: int, path="", destination_path="") -> pd.DataFrame:
#     """
#     Clusters the input data into `k` clusters based on personality traits and preferences.

#     Parameters:
#     k (int): Number of clusters
#     path (str): Path to the JSON file containing user data (e.g., 'users_prefs.json').
#     destination_path (str): Path to the JSON file where the cluster assignment will be saved.

#     Returns:
#     pd.DataFrame: DataFrame containing the cluster assignment for each data point.
#     """

#     try:
#         # if path != "":
#         #     with open(path, "r") as file:
#         #         json_data = json.load(file)
#         #         dummy_df = pd.DataFrame(json_data)

#         users = get_users_json()

#         dummy_df = pd.DataFrame(users)

#         dummy_df["Extraversion"] = dummy_df["extraverted"] - dummy_df["introverted"]
#         dummy_df["Intuition"] = dummy_df["intuitive"] - dummy_df["observant"]
#         dummy_df["Thinking"] = dummy_df["thinking"] - dummy_df["feeling"]
#         dummy_df["Judging"] = dummy_df["judging"] - dummy_df["prospecting"]
#         dummy_df["Assertiveness"] = dummy_df["assertive"] - dummy_df["turbulent"]

#         traits = ["Extraversion", "Intuition", "Thinking", "Judging", "Assertiveness"]

#         scaler = MinMaxScaler(feature_range=(-1, 1))
#         dummy_df[traits] = scaler.fit_transform(dummy_df[traits])

#         mlb = MultiLabelBinarizer()
#         preferences_encoded = mlb.fit_transform(dummy_df["preferences"])
#         preferences_df = pd.DataFrame(preferences_encoded, columns=mlb.classes_)  # type: ignore

#         dummy_df = pd.concat(
#             [dummy_df.reset_index(drop=True), preferences_df.reset_index(drop=True)],
#             axis=1,
#         )

#         X = dummy_df[traits].values

#         kmeans = KMeans(n_clusters=k, random_state=42)
#         clusters = kmeans.fit_predict(X)

#         dummy_df["Cluster"] = clusters
#         output = dummy_df.loc[:, ["Cluster"]]

#         clusters_json = []

#         for i in range(len(dummy_df)):
#             integer_cluster_value = int(dummy_df["Cluster"].iloc[i])
#             clusters_json.append({"id": i, "cluster": integer_cluster_value})

#         # Saving data to destination file
#         json.dump(clusters_json, open("dummy_io/ai/clustered_users.json", "w"))

#         # saving the model
#         joblib.dump(kmeans, "dummy_io/ai/model.pkl")

#         return pd.DataFrame(output)

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return pd.DataFrame()


# def get_tasks_json() -> List[Dict[str, Any]]:
#     tasks = get_tasks()
#     json_tasks = []
#     for task in tasks:
#         json_tasks.append({"name": task.name, "category": task.category})
#     return json_tasks


# def get_users_json(prefs=True) -> List[Dict[str, Any]]:
#     users = get_all_users()
#     json_users = []
#     for user in users:
#         if prefs:
#             json_users.append(
#                 {
#                     "id": user.id,
#                     "introverted": user.introverted,
#                     "extraverted": user.extraverted,
#                     "observant": user.observant,
#                     "intuitive": user.intuitive,
#                     "thinking": user.thinking,
#                     "feeling": user.feeling,
#                     "judging": user.judging,
#                     "prospecting": user.prospecting,
#                     "assertive": user.assertive,
#                     "turbulent": user.turbulent,
#                     "preferences": user.preferences,
#                 }
#             )
#         else:
#             json_users.append(
#                 {
#                     "id": user.id,
#                     "introverted": user.introverted,
#                     "extraverted": user.extraverted,
#                     "observant": user.observant,
#                     "intuitive": user.intuitive,
#                     "thinking": user.thinking,
#                     "feeling": user.feeling,
#                     "judging": user.judging,
#                     "prospecting": user.prospecting,
#                     "assertive": user.assertive,
#                     "turbulent": user.turbulent,
#                 }
#             )
#     return json_users


# def get_processed_tasks_users() -> List[Dict[str, Any]]:
#     users = get_tasked_users()
#     json_users = []
#     for user in users:
#         json_users.append({"id": user.id, "tasks": user.tasks})
#     return json_users


# def get_processed_clusters() -> List[Dict[str, Any]]:
#     clusters = get_clustered_users()
#     json_clusters = []
#     for cluster in clusters:
#         json_clusters.append({"id": cluster.id, "cluster": cluster.cluster})
#     return json_clusters


# def process_new_user(user: UserScores):
#     """
#     This function processes a new user's raw input into a feature vector by computing differences
#     between personality trait scores, and scales the resulting vector to a range of [-1, 1].

#     Input:
#         - user (list): A list representing the raw personality trait scores of the user.

#     Output:
#         - Returns a normalized feature vector (list) for the user after processing and scaling.
#     """
#     new_vector: list[float] = []

#     # Calculate the differences between paired personality traits.
#     extraversion = user.extraverted - user.introverted
#     intuition = user.intuitive - user.observant
#     thinking = user.thinking - user.feeling
#     judging = user.judging - user.prospecting
#     assertiveness = user.assertive - user.turbulent

#     new_vector.append(extraversion)
#     new_vector.append(intuition)
#     new_vector.append(thinking)
#     new_vector.append(judging)
#     new_vector.append(assertiveness)

#     # Scale the vector to be in the range of [-1, 1].
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     new_vector = scaler.fit_transform(np.array([new_vector]))[0]

#     return new_vector


# def cluster_new_user(user: np.ndarray):
#     """
#     This function loads a pre-trained KMeans model from a pickle file ('model.pkl') and uses it
#     to predict the cluster for a new user.

#     Input:
#         - user (np.ndarray): A 2D NumPy array representing the new user in terms of features.

#     Output:
#         - Returns the predicted cluster for the user as an integer.
#     """
#     kmeans = joblib.load("dummy_io/ai/model.pkl")
#     return kmeans.predict(user)


# def recommend_tasks(
#     user: np.ndarray, user_prefs: list, k: int = 5, work_end_time=17, sleep_time=24, work_start_time=8, user_id=-1
# ) -> tuple[dict[str, dict[str, str]], list[int]]:
#     """
#     This function recommends tasks to a user based on the similarity between the new user's
#     preferences and those of other users in the same cluster.

#     Input:
#         - user (np.ndarray): A 2D NumPy array representing the new user in terms of features.
#         - k (int): The number of similar users to consider for task recommendations.
#         - user_prefs (list): A list of the user's preferences.

#     Output:
#         - Returns a list of task names that are recommended for the user based on similarity
#           to other users and their preferences.
#     """

#     users_with_prefs = get_users_json()
#     users_without_prefs = get_users_json(prefs=False)
#     clusters = get_processed_clusters()
#     # Load necessary data from JSON files.
#     data = pd.DataFrame(clusters).drop("id", axis=1)
#     users_db = pd.DataFrame(users_without_prefs).drop("id", axis=1)
#     prefs_db = pd.DataFrame(users_with_prefs).drop("id", axis=1)
#     user_task_table = pd.DataFrame(get_processed_tasks_users()).drop("id", axis=1)
#     tasks_db = pd.DataFrame(get_tasks_json())

#     # Process the new user to transform their input features.
#     user_scores = UserScores(
#         extraverted=user[1],
#         introverted=user[0],
#         intuitive=user[3],
#         observant=user[2],
#         thinking=user[5],
#         feeling=user[4],
#         judging=user[7],
#         prospecting=user[6],
#         assertive=user[8],
#         turbulent=user[9],
#     )
#     processed_user = process_new_user(user_scores)

#     # Predict the cluster of the new user.
#     user_cluster = cluster_new_user(np.array([processed_user]))[0]

#     # Get users in the same cluster.
#     data = users_db.loc[data[data["cluster"] == user_cluster].index]
#     ids = data.index

#     user_array = np.array([user])
#     data_array = data.to_numpy()

#     # Compute similarity between the new user and users in the same cluster.
#     similarity_scores = cosine_similarity(user_array, data_array)
#     similar_ids = similarity_scores.argsort()[0][-k:]

    
#     # Find preferences of similar users.
#     similar_prefs = prefs_db.loc[similar_ids, "preferences"].values  # type: ignore
#     sets = [set(lst) for lst in similar_prefs]
#     given_set = set(user_prefs)
#     intersection_with_given_list = [s.intersection(given_set) for s in sets]
#     flat_intersection = set().union(*intersection_with_given_list)

#     # Find tasks that similar users have done.
#     similar_users_tasks = user_task_table.loc[similar_ids, "tasks"].values  # type: ignore
#     tasks_set = [set(lst) for lst in similar_users_tasks]
#     flat_tasks = set().union(*tasks_set)

#     # Filter tasks by matching the user's preference.
#     raw_tasks = tasks_db.loc[tasks_db["name"].isin(flat_tasks)]
#     tasks_for_users = raw_tasks.loc[raw_tasks["category"].isin(flat_intersection)]
#     tasks_to_recommend_with_categories = tasks_for_users.loc[:, ["name", "category"]].values
#     categories = list(tasks_for_users["category"].values)
#     tasks_to_recommend = [
#         {
#             "name": tasks_to_recommend_with_categories[i][0],
#             "category": tasks_to_recommend_with_categories[i][1],
#         }
#         for i in range(len(tasks_to_recommend_with_categories))
#     ]


    

#     top_four = tasks_to_recommend[:4]
#     schedule = get_tasks_last_n_weeks(1, categories, ids, top_four, 13, sleep_time)
#     filtered_schedule = [
#         {
#             "name": key,
#             "startTime": value,
#             "endTime": f"{int(value[:2]) + 1:02}:00"
#         }
#         for key, value in schedule.items()
#     ]

#     filtered_schedule.insert(0, {
#         "name":"blocked",
#         "startTime": f"{f"0{work_start_time}" if work_start_time < 10 else f"{work_start_time}"}:00",
#         "endTime": f"{f"0{work_end_time}" if work_end_time < 10 else f"{work_end_time}"}:00"
#     })

#     if user_id != -1:
#         similar_ids = list(similar_ids)
#         similar_ids.append(user_id)
#     today = datetime.datetime.today()
#     day_name = today.strftime("%A")
#     response = {
#         "day": day_name,
#         "tasks": filtered_schedule
#     }
#     return response, [int(i) for i in similar_ids]

# def get_tasks_last_n_weeks(n: int, categories:List[str], user_ids: List[int], new_tasks: List[ Dict[str, Any]], work_end_time=17, sleep_time=24) -> Dict[str, str]:
    

#     try:
#         schedules = get_schedules(SchedulesRequest(
#         weeks= [i for i in range(n+1)]
#         ))
#         today = datetime.datetime.today()
#         day_name = today.strftime("%A")
#         relevant_schedules = [
#             task for s in schedules
#             if int(s["user"]) in user_ids # type: ignore
#             for t in s['schedules']
#             if t['day'] == day_name # type: ignore
#             for task in t['tasks'] # type: ignore
#             if task['category'] in categories # type: ignore
#             and task["completed"] # type: ignore
#         ]


#         print(relevant_schedules)

#         model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#         # similar_tasks = get_top_n_similar_tasks(new_task, relevant_schedules, model)
#         # predicted_start_time = weighted_average_start_time(similar_tasks)
#         # print(f"The recommended start time for '{new_task['name']}' is: {predicted_start_time}")
#         scheduled_times = recommend_task_start_times(new_tasks, relevant_schedules, work_end_time, sleep_time, model) # type: ignore
#         for task_name, start_time in scheduled_times.items():
#             print(f"The recommended start time for '{task_name}' is: {start_time}")
#         return scheduled_times

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return {}

# def get_name_similarity(task1: Dict[str, Any], task2: Dict[str, Any], model) -> float:
#     """
#     Calculate the cosine similarity between the names of two tasks.

#     Args:
#         task1 (dict): First task with at least a 'name' key.
#         task2 (dict): Second task with at least a 'name' key.
#         model: Model with an 'encode' method to generate embeddings.

#     Returns:
#         float: Cosine similarity between the task names.
#     """
#     try:
#         task1_name = task1['name']
#         task2_name = task2['name']
#     except KeyError as e:
#         raise KeyError(f"Missing 'name' key in task: {e}")

#     task1_embedding = model.encode(task1_name)
#     task2_embedding = model.encode(task2_name)

#     # Ensure embeddings are 2D arrays
#     task1_embedding = np.atleast_2d(task1_embedding)
#     task2_embedding = np.atleast_2d(task2_embedding)

#     similarity = cosine_similarity(task1_embedding, task2_embedding)[0][0]

#     return similarity

# def get_top_n_similar_tasks(
#     new_task: Dict[str, Any],
#     task_list: List[Dict[str, Any]],
#     model,
#     n: int = 5
# ) -> List[Tuple[Dict[str, Any], float]]:
#     """
#     Get the top N most similar tasks based on name similarity.

#     Args:
#         new_task (dict): The new task to compare.
#         task_list (list): List of existing tasks.
#         model: Model with an 'encode' method to generate embeddings.
#         n (int): Number of top similar tasks to return.

#     Returns:
#         list: List of tuples (task, similarity_score), sorted by similarity.
#     """
#     if not task_list:
#         return []

#     task_similarities = []
#     for task in task_list:
#         try:
#             similarity = get_name_similarity(new_task, task, model)
#             task_similarities.append((task, similarity))
#         except KeyError:
#             continue  # Skip tasks without 'name' key

#     sorted_tasks = sorted(task_similarities, key=lambda x: x[1], reverse=True)

#     return sorted_tasks[:n]

# def weighted_average_start_time(similar_tasks: List[Tuple[Dict[str, Any], float]]) -> int:
#     """
#     Calculate the weighted average start time based on similar tasks.

#     Args:
#         similar_tasks (list): List of tuples (task, similarity_score).

#     Returns:
#         int: Weighted average start time in hours (integer).
#     """
#     weighted_times = []
#     total_weight = 0

#     for task, similarity in similar_tasks:
#         try:
#             start_time_str = task['startTime']  # Expected format 'HH:MM'
#             start_time = datetime.datetime.strptime(start_time_str, '%H:%M')
#             start_time_in_minutes = start_time.hour * 60 + start_time.minute
#             weighted_times.append(start_time_in_minutes * similarity)
#             total_weight += similarity
#         except (KeyError, ValueError):
#             continue  

#     if not weighted_times:
#         return 20  

#     if total_weight > 0:
#         weighted_avg_time_in_minutes = sum(weighted_times) / total_weight
#     else:
#         weighted_avg_time_in_minutes = sum(weighted_times) / len(weighted_times)

#     weighted_avg_time_in_hours = round(weighted_avg_time_in_minutes / 60)

#     return weighted_avg_time_in_hours

# def calculate_category_frequency(existing_tasks: List[Dict[str, Any]], work_end_time: int, sleep_time: int) -> Dict[str, Dict[int, int]]:
#     """
#     Calculate the frequency of tasks in each category within hourly time slots.

#     Args:
#         existing_tasks (list): List of existing tasks.
#         work_end_time (int): Hour when work ends.
#         sleep_time (int): Hour when sleep time starts.

#     Returns:
#         dict: A nested dictionary where keys are categories and values are dictionaries
#               mapping hour slots to frequency counts.
#     """
#     category_frequency = defaultdict(lambda: defaultdict(int))

#     for task in existing_tasks:
#         try:
#             start_hour = int(task['startTime'].split(":")[0])
#             if work_end_time <= start_hour < sleep_time:
#                 category_frequency[task['category']][start_hour] += 1
#         except (KeyError, ValueError):
#             continue  # Skip tasks with missing or invalid start times

#     return category_frequency # type: ignore

# def recommend_task_start_times(
#     new_tasks: List[Dict[str, Any]],
#     existing_tasks: List[Dict[str, Any]],
#     work_end_time: int,
#     sleep_time: int,
#     model,
#     n: int = 5,
#     time_gap: int = 1,
#     similarity_threshold: float = 0.5
# ) -> Dict[str, str]:
#     """
#     Recommend start times for new tasks based on category frequency in specific time slots.

#     Args:
#         new_tasks (list): List of new tasks.
#         existing_tasks (list): List of existing tasks.
#         work_end_time (int): Hour when work ends (e.g., 17 for 5 PM).
#         sleep_time (int): Hour when sleep time starts (e.g., 23 for 11 PM).
#         model: Model with an 'encode' method to generate embeddings.
#         n (int): Number of top similar tasks to consider.
#         time_gap (int): Required gap between tasks in hours for dissimilar tasks.
#         similarity_threshold (float): Threshold above which tasks are considered similar.

#     Returns:
#         dict: Mapping from task names to scheduled start times in 'HH:MM' format.
#     """
#     # Step 1: Calculate category frequency in time slots
#     category_frequency = calculate_category_frequency(existing_tasks, work_end_time, sleep_time)

#     available_times = []
#     for hour in range(work_end_time, sleep_time):
#         available_times.append(datetime.datetime.strptime(f"{hour}:00", "%H:%M"))
#     available_times.sort()

#     scheduled_tasks = {}
#     last_scheduled_time = None
#     last_task_name = None

#     for new_task in new_tasks:
#         category = new_task.get('category', None)

#         if category in category_frequency:
#             # Step 2: Find the peak hour for the category
#             peak_hour = max(category_frequency[category], key=category_frequency[category].get) # type: ignore
#             recommended_time = datetime.datetime.strptime(f"{peak_hour}:00", "%H:%M")
#         else:
#             recommended_time = available_times[0]  # Default to earliest available if no history

#         if last_scheduled_time is None:
#             earliest_start_time = available_times[0]
#         else:
#             last_task = {'name': last_task_name}
#             similarity_with_last = get_name_similarity(new_task, last_task, model)

#             if similarity_with_last >= similarity_threshold:
#                 earliest_start_time = last_scheduled_time
#             else:
#                 earliest_start_time = last_scheduled_time + datetime.timedelta(hours=time_gap)

#         # Step 3: Find the closest available time to the recommended time
#         potential_times = [t for t in available_times if t >= earliest_start_time]
#         if not potential_times:
#             break

#         scheduled_time = min(
#             potential_times,
#             key=lambda t: abs((t - recommended_time).total_seconds())
#         )

#         scheduled_tasks[new_task['name']] = scheduled_time.strftime('%H:%M')

#         # Update available times by removing the scheduled time slot
#         task_duration = datetime.timedelta(hours=1)
#         end_time = scheduled_time + task_duration
#         available_times = [t for t in available_times if t < scheduled_time or t >= end_time]

#         last_scheduled_time = end_time
#         last_task_name = new_task['name']

#     return scheduled_tasks



# def main():
#     new_user = np.array([0.0, 5.0, 0.0, 31.0, 0.0, 19.0, 0.0, 22.0, 44.0, 0.0])
#     user_prefs = [
#         "Health",
#         "Exersice",
#         "Learning",
#         "Finance",
#         "Creative",
#         "Exercise",
#         "Mental Wellness",
#     ]
#     k = 5

#     print(recommend_tasks(new_user, user_prefs, k))


# if __name__ == "__main__":
#     main()
