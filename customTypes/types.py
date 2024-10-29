

from pydantic import BaseModel

class UserProfileRequest(BaseModel):
    ids: list[int]
    req_scores: bool
    req_preferences: bool


class SchedulesRequest(BaseModel):
    weeks: list[int]



class UserScores(BaseModel):
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


# User profile returned to AI
class UserProfile(BaseModel):
    id: int
    scores: UserScores
    preferences: list[str]

class ScheduledTask(BaseModel):
    name: str
    category: str
    startTime:str
    endTime:str
    completed: bool

class DailySchedule(BaseModel):
    day: str
    tasks: list[ScheduledTask]


class Recommendation(BaseModel):
    tasks: dict[str, int]


class RecommendationInput(BaseModel):
    scores: list[float]
    preferences: list[str]
    work_end_time: int
    sleep_time: int
    user_id: int


class UserTask(BaseModel):
    id: int
    tasks: list[str]


class Task(BaseModel):
    name: str
    category: str


class ClusteredUsers(BaseModel):
    id: int
    cluster: int


# User profile returned to AI
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
    preferences: list[str]



