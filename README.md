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
│   ├── llm.py           # Language model integration
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
- PostgreSQL database
- OpenAI API key

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
The backend communicates with:
- NeonDB for data storage
- OpenAI API for task generation
- AWS Lambda functions for schedule management

### Database Schema
Uses PostgreSQL with the following main tables:
- users
- profiles
- schedules
- tasks

## Deployment

The application is designed to be deployed as a containerized service.

### Docker Deployment
1. Build the image:
```bash
docker build -t bustleit-ai .
```

2. Run the container:
```bash
docker run -p 5120:5120 bustleit-ai
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, email [support@bustleit.com](mailto:support@bustleit.com) or create an issue in the repository.
