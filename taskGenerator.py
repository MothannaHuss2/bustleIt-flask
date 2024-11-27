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

load_dotenv()

OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")

categories = [
    'Sports',
    'Gaming',
    'Reading',
    'Nightlife',
    'Study' 
]
class Task(BaseModel):
    name: str = Field(description="The name of the task")
    category: str = Field(description="The category of the task")
class Output(BaseModel):
    tasks: list[Task] = Field(description="The tasks that the AI will recommend")

@lru_cache(maxsize=200) 
def generator(category):
    try:
        llm = ChatOpenAI(model="gpt-4o", api_key=OPEN_AI_KEY)
        prompt = f"Generate a list of tasks in the {category} category"
        
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

def main():
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
    with open('tasks.json', 'w') as f:
        f.write(json_tasks)
        logger.info('Tasks written to tasks.json')
        
main()
    
    