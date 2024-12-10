from dotenv import load_dotenv
import os
import requests
import json
from logger import get_logger
from customTypes import ClusteredProfile, BatchedTasks, Schedule, DailyTask,RawProfile
from datetime import datetime
load_dotenv()
logger = get_logger('External API')

token = 'b15cJ3iOc8tZRfJaoBLUg3oXXBk25Ov9kfZKc0yL'
api_url = 'https://yz69xl3axh.execute-api.eu-central-1.amazonaws.com/Prod'



    

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
        logger.info('Error at api.getUsersByCluster:', e)
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
        logger.info("Error at api.getUserById " + e)
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
        logger.info(f'Error at getUsersByIds: {e}')
        return []



def getAllTasks() -> list[BatchedTasks]:
    try:
        url = f"{api_url}/v1/tasks"
        header = {
            'Authorization': f'Bearer {token}'
        }
        response = requests.get(url, headers=header)
        tasks = response.json()
        casted_tasks = []
        for task in tasks:
            casted= BatchedTasks(user_id=task['user_id'], all_tasks=task['all_tasks'])
            for t in casted.all_tasks:
                t.start_time = datetime.strptime(t.start_time, "%Y-%m-%d %H:%M:%S UTC").strftime("%H:%M")
                t.end_time = datetime.strptime(t.end_time, "%Y-%m-%d %H:%M:%S UTC").strftime("%H:%M")
                t.created_at = datetime.strptime(t.created_at.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S.%f")
                t.updated_at = datetime.strptime(t.updated_at.replace(" UTC", ""), "%Y-%m-%d %H:%M:%S.%f")
            casted_tasks.append(casted)
        return casted_tasks
    except Exception as e:
        logger.info(f'Erorr at getAllTasks: {e}')
        return []
def getAllUsers() -> list[ClusteredProfile]:
    try:
        url = f"{api_url}/v1/user/profiles"
        headers= {
            "Authorization": f"Bearer {token}"
        }
        response = requests.get(url, headers=headers)
        users = response.json()
        casted = [RawProfile(user_id=user['user_id'], scores=user['scores'], preferences=user['preferences']) for user in users]
        return casted
    except Exception as e:
        logger.info('Error at api.getAllUsers:', e)
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
        logger.info("Error at api.getUserSchedule " + e)
        return None



def getAllUsersTasks() -> list[DailyTask]:
    try:
        url = f'{api_url}/v1/tasks'
        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = requests.get(url, headers=headers)
        tasks = response.json()
        excluded = [ 'Finance', 'Career','Work','Personal Development']
        returnUs = []
        for task in tasks:
            all_tasks = task['all_tasks']
            for t in all_tasks:
                if not t['category'] in excluded:
                    t['start_time'] = datetime.strptime(t['start_time'], "%Y-%m-%d %H:%M:%S UTC").strftime("%H:%M").__str__()
                    t['end_time'] = datetime.strptime(t['end_time'], "%Y-%m-%d %H:%M:%S UTC").strftime("%H:%M").__str__()
                    returnUs.append(
                        DailyTask(
                        task_id=t['task_id'],
                        name=t['name'],
                        category=t['category'],
                        start_time=t['start_time'],
                        end_time=t['end_time'],
                        completed=t['completed'],
                        created_at=t['created_at'].strip().replace(" UTC", "").replace(" ", "T").split(".")[0],
                        updated_at=t['updated_at'].strip().replace(" UTC", "").replace(" ", "T").split(".")[0]
                    ))
        return returnUs
    except Exception as e:
        logger.info(f'Error at getAllUsersTasks: {e}')
        return []

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
            # logger.info(f'User Schedule {task["user_id"]} has {len(task["all_tasks"])} tasks')
            totalTasks += len(task["all_tasks"])
            casted_task = BatchedTasks(user_id=task['user_id'], all_tasks=task['all_tasks'])
            for t in casted_task.all_tasks:
                t.start_time = datetime.strptime(t.start_time, "%Y-%m-%d %H:%M:%S UTC").strftime("%H:%M")
                t.end_time = datetime.strptime(t.end_time, "%Y-%m-%d %H:%M:%S UTC").strftime("%H:%M")
                t.created_at = t.created_at.strip().replace(" UTC", "").replace(" ", "T").split(".")[0]
                t.updated_at = t.updated_at.strip().replace(" UTC", "").replace(" ", "T").split(".")[0]
            casted.append(
                casted_task
            )
        
        return casted
    except Exception as e:
        logger.info(f"Error in getBatchedTasks " + str(e))
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
        logger.info(f'Error at getRangedSchedule: {e}')
        return None