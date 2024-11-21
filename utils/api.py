from dotenv import load_dotenv
import os
import requests
from pydantic import BaseModel
from typing import List, Dict
import json
from logger import get_logger
load_dotenv()
logger = get_logger('External API')

token = os.getenv("API_TOKEN")
api_url = os.getenv("BASE_URI")

class ClusteredProfile(BaseModel):
    user_id: str
    scores: Dict[str, float]
    preferences: List[str]
    cluster: int
    
class DailyTask(BaseModel):
    task_id: str
    name: str
    category: str
    start_time: str
    end_time: str
    completed: bool
    created_at: str
    updated_at: str
    
class DailySchedule(BaseModel):
    total_tasks: int
    completed_tasks: int
    tasks: List[DailyTask]
    
class Schedule(BaseModel):
    user_id: str
    data: Dict[str, DailySchedule]
class BatchedTasks(BaseModel):
    user_id: str
    all_tasks: List[DailyTask]
    

    

def getUsersByCluster(cluster) -> list[ClusteredProfile]:
    try:
        url = f"{api_url}/v1/user/profiles?cluster={cluster}"
        headers= {
            "Authorization": f"Bearer {token}"
        }
        
        response = requests.get(url, headers=headers)
        users = response.json()
        casted = [ClusteredProfile(user_id=user['user_id'], scores=user['scores'], preferences=user['preferences'], cluster=cluster) for user in users]
        return casted
    except Exception as e:
        print('Error:', e)
        return []



def getUserById(id: str) -> ClusteredProfile:
    try:
        url = f'{api_url}/v1/user/profile/{id}'
        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = requests.get(url, headers=headers)
        user = response.json()
        print(user)
        casted = ClusteredProfile(user_id=user['user_id'], scores=user['scores'], preferences=user['preferences'], cluster=1)
        return casted
    except Exception as e:
        print("Error " + e)
        return None

def getUsersByIds(ids):
    try:
        url= f'{api_url}/v1/user/profiles/batch'
        headers= {
            "Authorization": f"Bearer {token}",
            "content-type": "application/json"
        }
        body = {
            "user_ids": ids
        }
        response = requests.post(url, headers=headers, json=body)
        users = response.json()
        return users
    except Exception as e:
        print('Error:', e)
        return []


def getAllUsers() -> list[ClusteredProfile]:
    try:
        url = f"{api_url}/v1/user/profiles"
        headers= {
            "Authorization": f"Bearer {token}"
        }
        response = requests.get(url, headers=headers)
        users = response.json()
        casted = [ClusteredProfile(user_id=user['user_id'], scores=user['scores'], preferences=user['preferences']) for user in users]
        return casted
    except Exception as e:
        print('Error:', e)
        return []

def getUserSchedule(id: str) -> Schedule:
    try:
        url = f'{api_url}/v1/user/{id}/schedule'
        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = requests.get(url, headers=headers)
        schedule = response.json()
        casted = Schedule(user_id=schedule['user_id'], data=schedule['data'])
        return casted
    except Exception as e:
        print("Error " + e)
        return None



def getBatchedTasks(ids: list[str]) -> list[BatchedTasks]:
    try:
        url = f'{api_url}/v1/tasks/batch'
        headers = {
            "Authorization": f"Bearer {token}",
            "content-type": "application/json"
        }
        body = {
            "user_ids": ids
        }
        response = requests.post(url, headers=headers, json=body)
        tasks = response.json()
       
        casted = []
        totalTasks = 0
        for task in tasks:
            logger.info(f'User Schedule {task["user_id"]} has {len(task["all_tasks"])} tasks')
            totalTasks += len(task["all_tasks"])
            casted.append(
                BatchedTasks(user_id=task['user_id'], all_tasks=task['all_tasks'])
            )
        logger.info(f'Total tasks fetched: {totalTasks}')
        
        return casted
    except Exception as e:
        print("Error in getBatchedTasks " + e.__str__())
        return []

def getRangedSchedule(id:str, startDate:str, range: int) -> Schedule:

    try:
        url = f'{api_url}/v1/user/{id}/schedule?date={startDate}&range={range}&skip_empty=true'
        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = requests.get(url, headers=headers)
        schedule = response.json()

        if not schedule:
            return None
        casted = Schedule(user_id=schedule['user_id'], data=schedule['data'])
        return casted
    except Exception as e:
        print("Error " + e)
        return None