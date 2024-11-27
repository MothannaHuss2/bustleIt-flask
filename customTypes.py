
from pydantic import BaseModel
from typing import List, Dict
class ClusteredProfile(BaseModel):
    user_id: str
    scores: Dict[str, float]
    preferences: List[str]
    cluster: int
class RawProfile(BaseModel):
    user_id: str
    scores: Dict[str, float]
    preferences: List[str]
    
    
class DailyTaskInput(BaseModel):
    user_id: str
    scores: Dict[str, float]
    preferences: List[str]
    cluster: int
    work_end_time: int
    sleep_time: int
    work_start_time: int
class DailyTask(BaseModel):
    task_id: str
    name: str
    category: str
    start_time: str
    end_time: str
    completed: bool
    created_at: str
    updated_at: str
    
class WeeklyRecommendationInput(BaseModel):
    user_id: str
    scores: Dict[str, float]
    preferences: List[str]
    cluster: int
    work_end_time: int
    sleep_time: int
    work_start_time: int

    
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
    