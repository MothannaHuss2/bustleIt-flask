# BustleIt AI Backend

A Flask-based AI backend service for task recommendation and user clustering.

## Overview

The BustleIt AI Backend is responsible for:
- Processing and analyzing user personality data
- Clustering users based on personality traits and preferences 
- Generating personalized task recommendations
- Managing daily and weekly schedule recommendations

## Project Structure

```
bustleIt-flask-main/
├── tasks.py              # Task definitions and data
├── config.py             # Configuration settings
├── utils/
│   └── api.py           # External API communication
├── models/              
│   └── model.pkl        # ML model files
├── logger.py            # Logging configuration
├── customTypes.py       # Type definitions
├── ai/
│   └── ai_cluster.py    # Clustering algorithms
├── app.py              # Main Flask application
└── taskGenerator.py    # Task generation utilities
```

## Core Features

### User Clustering
- Implements K-means clustering algorithm
- Groups users based on:
  - Personality scores (introversion/extroversion, etc.)
  - User preferences
  - Activity patterns

### Task Recommendation
- Generates personalized daily and weekly schedules
- Considers:
  - User's work schedule
  - Sleep patterns
  - Activity preferences
  - Historical task completion rates

### Schedule Generation
- Creates balanced activity schedules
- Handles:
  - Time slot allocation
  - Task variety
  - Activity pacing
  - Schedule conflicts

## API Endpoints

### Clustering
- `POST /cluster`: Assigns a user to a cluster
- `POST /rank`: Ranks similar users within a cluster

### Recommendations
- `POST /recommend_daily`: Generates daily schedule recommendations
- `POST /recommend_weekly`: Generates weekly schedule recommendations

## Setup

### Prerequisites
- Python 3.8+

### Environment Variables
```
OPENAI_API_KEY=your_openai_key
API_TOKEN=your_api_token
BASE_URI=your_base_uri
```

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd bustleIt-flask-main
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

### Running the Application

1. Start the Flask server:
```bash
python app.py
```

The server will start on port 5120 by default.

## Development

### Adding New Tasks
1. Edit `tasks.py` to add new task definitions
2. Run `taskGenerator.py` to regenerate task data:
```bash
python taskGenerator.py
```

### Model Retraining
To retrain the clustering model:
1. Collect updated user data
2. Call the retraining endpoint:
```bash
curl -X POST http://localhost:5120/retrain_model
```

### Testing
Run the test suite:
```bash
python -m pytest tests/
```

## Integration

### External API Integration

The backend communicates with several external services:

#### NeonDB Integration
- Handles all persistent data storage
- Uses PostgreSQL compatible connection
- Manages user profiles, tasks, and schedules

#### OpenAI API Integration
- Used for task generation and recommendations
- Model: GPT-4
- Implements retry logic and error handling
- Request format:
```python
messages = [
    SystemMessage(content=prompt),
    HumanMessage(content=f'Tasks: {tasks}\nWork end time: {work_end_time}\nSleep time: {sleep_time}')
]
```

#### External API Communication

##### Base Configuration
```python
token = os.getenv("API_TOKEN")
api_url = os.getenv("BASE_URI")
```

##### Available Endpoints

1. User Profile Management:
```python
# Get users by cluster
GET /v1/user/profiles?cluster={cluster}
Headers: Authorization: Bearer {token}

# Get user by ID
GET /v1/user/profile/{id}
Headers: Authorization: Bearer {token}

# Get batch user profiles
POST /v1/user/profiles/batch
Headers: 
  - Authorization: Bearer {token}
  - Content-Type: application/json
Body: {
    "user_ids": ["uuid1", "uuid2"]
}
```

2. Task Management:
```python
# Get all tasks
GET /v1/tasks
Headers: Authorization: Bearer {token}

# Get batch tasks
POST /v1/tasks/batch
Headers: 
  - Authorization: Bearer {token}
  - Content-Type: application/json
Body: {
    "user_ids": ["uuid1", "uuid2"]
}

# Get user schedule
GET /v1/user/{user_id}/schedule?date={date}&range={range}&skip_empty={boolean}
Headers: Authorization: Bearer {token}
```

3. Schedule Generation:
```python
# Get daily recommendation
POST /recommend_daily
Headers: Content-Type: application/json
Body: {
    "user_id": "uuid",
    "scores": {
        "introverted": float,
        "extraverted": float,
        ...
    },
    "preferences": ["pref1", "pref2"],
    "cluster": int,
    "work_end_time": int,
    "sleep_time": int,
    "work_start_time": int
}

# Get weekly recommendation
POST /recommend_weekly
Body: same as daily recommendation
```

##### Error Handling
```python
try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()
except requests.exceptions.RequestException as e:
    logger.error(f"API request failed: {e}")
    return None
```

##### Response Format Examples

1. User Profile Response:
```json
{
    "user_id": "uuid",
    "scores": {
        "introverted": 0.7,
        "extraverted": 0.3,
        ...
    },
    "preferences": ["Sports", "Reading"],
    "cluster": 1
}
```

2. Schedule Response:
```json
{
    "user_id": "uuid",
    "data": {
        "2024-12-20": {
            "total_tasks": 4,
            "completed_tasks": 2,
            "tasks": [
                {
                    "task_id": "uuid",
                    "name": "Task Name",
                    "category": "Category",
                    "start_time": "HH:MM",
                    "end_time": "HH:MM",
                    "completed": false
                }
            ]
        }
    }
}
```

### Database Schema
Uses PostgreSQL with the following main tables:
- users
- profiles
- schedules
- tasks

## Deployment

The application is designed to be deployed as a containerized service.


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
