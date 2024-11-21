from flask import Flask, render_template
import joblib
from flask import Flask, request, jsonify
from customTypes.types import RecommendationInput, Recommendation, RawProfile
import utils.api as api
import ai.ai_cluster as AI
app = Flask(__name__)






# Define the home route
@app.route('/')
def home():
    id = '542172eb-c417-46c0-b9b1-78d1b7630bf5'
    user = api.getUserById(id)
    print(user)
    sims = AI.recommend_weekly_tasks(user)
    print(sims)
    return sims

# Define additional routes (example route)
@app.route('/about')
def about():
    return "This is the About Page."



@app.post("/cluster")
def cluster():
    try:
        profile = RawProfile(**request.json)
        cluster = int(AI.cluster_single_record(profile))
        return jsonify({"cluster": cluster})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.post('/recommend')
def recommend():
    # Parse and validate the request body
    try:
        recommendation_input = RecommendationInput(**request.json)
    except ValueError as e:
        # Return validation error if the data is invalid
        return jsonify({"error": str(e)}), 400
    recommended_tasks = {"task1": 1, "task2": 2}  # Example dictionary with task names and priority/importance

    recommendation = Recommendation(tasks=recommended_tasks)

    return jsonify(recommendation.dict())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5120, debug=True)

