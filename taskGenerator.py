from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
import time
from functools import lru_cache
from logger import get_logger
logger = get_logger(__name__)
from dotenv import load_dotenv
import json
import os
import random
load_dotenv()

OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

categories = [
    'football',
    'pets',
    'gaming',
    'weight loss',
    'basketball',
    'cooking',
    'tech',
    'programming',
    'books',
    'art',
    'languages',
    'volunteering'
]
class Task(BaseModel):
    name: str = Field(description="The name of the task")
    category: str = Field(description="The category of the task")
class Output(BaseModel):
    tasks: list[Task] = Field(description="The tasks that the AI will recommend")
class ScheduledTasks(BaseModel):
    name: str = Field(description="The name of the task")
    category: str = Field(description="The category of the task")
    startTime: str = Field(description="The time at which the task starts")
    endTime: str = Field(description="The time at which the task ends")
class Schedule(BaseModel):
    schedule: list[ScheduledTasks] = Field(description="All the tasks in the schedule")

def scheduleGenerator(tasks):
    try:
        llm = ChatOpenAI(model="gpt-4o", api_key=OPEN_AI_KEY)
        prompt = f'Generate a schedule out of those tasks: {tasks}, make sure you place them in a cohesive manner. You can assume that each one takes 1 hour to finish, and they should start at least after 5 PM, use 24 hour time to set the start and end times and get each task to start and end at an exact hour like 15:00, 16:00 no portions.'
        llm = llm.with_structured_output(Schedule)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f'Those are the tasks {tasks}'),
        ]
        output = llm.invoke(messages)
        return output
    except Exception as e:
        logger.info('Error at scheduler: ', e)
        return []
        
def generator(category):
    try:
        llm = ChatOpenAI(model="gpt-4o", api_key=OPEN_AI_KEY)
        prompt = f"Generate a list of tasks in the {category} category, strictly set its category to {category}. You are generating tasks for people to do off work"
        
        llm = llm.with_structured_output(Output)
        
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f'Generate 40 tasks in the {category} category'),
            ]
        output = llm.invoke(messages)
        return output
    except Exception as e:
        logger.info('Error at generator:', e)
        return []

def write():
    all_tasks = [
        
    ]
    for category in categories:
        output = generator(category)
        for task in output.tasks:
            all_tasks.append({
                'name': task.name,
                'category': task.category
            })
        time.sleep(15)
        logger.info(f'\nGenerated tasks for {category} are: \n{output}\n')
    
    json_tasks = json.dumps(all_tasks)
    with open('tasks1.json', 'w') as f:
        f.write(json_tasks)
        logger.info('Tasks written to tasks.json')
def scheduler():
    schedules = []
    with open('tasks1.json', 'r') as file:
        loaded = json.load(file)
        tasks = [task for task in loaded]
        for i in range(30):
            day = i+1
            for j in range(5):
                tasks_per_day = random.randint(3,6)
                sample = random.sample(tasks, tasks_per_day)
                schedule = scheduleGenerator(sample)
                time.sleep(3)
                s= []
                completed = 0
                for task in schedule.schedule:
                    value = random.randint(0,1)
                    if value > 0:
                        completed +=1
                    obj = {
                        'name':task.name,
                        'category': task.category,
                        'start_time': f'2024-11-{day if day > 9 else f'{day}0'} {task.startTime}:00 UTC',
                        'end_time': f'2024-11-{day if day > 9 else f'{day}0'} {task.endTime}:00 UTC',
                        'completed': True if value == 1 else False,
                    }
                    s.append(obj)
                
                schedules.append({
                    'id':f'{j}_{day}',
                    f'2024-11-{day if day > 9 else f'{day}0'}':
                        {
                            'total_tasks':len(s),
                            'completed_tasks': completed,
                            'tasks':s
                        }
                })
    
    json_schedules = json.dumps(schedules)
    with open('schedules.json', 'w') as file:
        file.write(json_schedules)
def main():
    updated = []
    with open('schedules.json','r') as file:
        jsonF = json.load(file)
        for s in jsonF:
            
            updated.append(s)
    print(len([u for u in updated if '1_' in u['id']]))
    # scheduler()
            
            
    
        
main()
    
    