from dotenv import load_dotenv
import os
import requests
from pydantic import BaseModel
from typing import List, Dict

load_dotenv()

token = os.getenv("API_TOKEN")
api_url = os.getenv("BASE_URI")

class ClusteredProfile(BaseModel):
    user_id: str
    scores: Dict[str, float]
    preferences: List[str]
    

def getUsersByCluster(cluster) -> list[ClusteredProfile]:
    try:
        url = f"{api_url}/v1/user/profiles?cluster={cluster}"
        headers= {
            "Authorization": f"Bearer {token}"
        }
        
        response = requests.get(url, headers=headers)
        users = response.json()
        return users
    except Exception as e:
        print('Error:', e)
        return []



def getUserById(id):
    try:
        url = f'{api_url}/v1/user/profile/{id}'
        headers = {
            "Authorization": f"Bearer {token}"
        }
        response = requests.get(url, headers=headers)
        user = response.json()
        return user
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
        print(len(users))
        print(users[0])
        casted = [ClusteredProfile(user_id=user['user_id'], scores=user['scores'], preferences=user['preferences']) for user in users]
        print(len(casted))
        return casted
    except Exception as e:
        print('Error:', e)
        return []