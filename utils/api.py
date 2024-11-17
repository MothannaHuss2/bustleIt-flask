from dotenv import load_dotenv
import os
import requests

load_dotenv()

token = os.getenv("API_TOKEN")
api_url = os.getenv("BASE_URI")

def getUsersByCluster(cluster):
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


def getAllUsers():
    try:
        url = f"{api_url}/v1/user/profiles"
        headers= {
            "Authorization": f"Bearer {token}"
        }
        response = requests.get(url, headers=headers)
        users = response.json()
        return users
    except Exception as e:
        print('Error:', e)
        return []