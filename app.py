from flask import Flask, render_template
import joblib
from flask import Flask, request, jsonify
from customTypes.types import RecommendationInput, Recommendation, RawProfile
from ai.ai_cluster import cluster_single_record
app = Flask(__name__)


# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define additional routes (example route)
@app.route('/about')
def about():
    return "This is the About Page."



@app.post("/cluster")
def cluster():
    try:
        profile = RawProfile(**request.json)
        cluster = int(cluster_single_record(profile))
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

    # Create a Recommendation instance
    recommendation = Recommendation(tasks=recommended_tasks)

    # Return the recommendation as JSON
    return jsonify(recommendation.dict())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5120, debug=True)

