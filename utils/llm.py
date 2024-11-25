from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.agents import AgentExecutor, create_openai_functions_agent
from logger import get_logger
logger = get_logger(__name__)
from dotenv import load_dotenv
import os
load_dotenv()

OPEN_AI_KEY = os.getenv("OPENAI_API_KEY")
class Task(BaseModel):
    name: str = Field(description="The name of the task")
    start_time: str = Field(description="The start time of the task")
    end_time: str = Field(description="The end time of the task")
class Recommendation(BaseModel):
    tasks: list[Task] = Field(description="The tasks that the AI will recommend")


def chain(tasks, work_end_time, sleep_time) -> Recommendation:
    try:
        chat = ChatOpenAI(model="gpt-4o", api_key=OPEN_AI_KEY)
        prompt = "You will be given a list of tasks, recommend four diverse tasks for a user to complete after work" + "\n You should make sure that tasks are ordered in a logical manner and they are diverse in nature"
        "Remember the user is an employee, tasks should not be too time-consuming or too difficult, but should be engaging and fulfilling"
        "Similar tasks should be grouped together, and the tasks should be ordered in a way that makes sense"
        "Users work end time and sleep time are gonna be given to you, so you can recommend tasks that can be done after work"
        "IMPORTANT: All tasks you recommend should take exactly one hour to complete, and should be engaging and fulfilling"
        "Example of a returned task: {name: 'Task name', start_time: '12:00', end_time: '13:00'}"
        "Important:"
        "When recommending a task that is irrelevant to its predecessor, make sure you include one hour time gap between them"
        ""
        llm = chat.with_structured_output(Recommendation)
        messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f'Tasks: {tasks}\nWork end time: {work_end_time + 1}\nSleep time: {sleep_time} every task should take one hour and make sure unrelated tasks have a one hour gap between them'),
        ]
        output = llm.invoke(messages)
        
        return output
    except Exception as e:
        logger.info('Error at chain:', e)
        return []