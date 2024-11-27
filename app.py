from flask import Flask, render_template
from flask import Flask, request, jsonify
import utils.api as api
import ai.ai_cluster as AI
from customTypes import RawProfile , WeeklyRecommendationInput , DailyTaskInput
import re

app = Flask(__name__)

# Define the home route
# @app.route('/')
# def home():
#     id = '00b4fb64-5b6e-4fe4-905b-b2ea7bfc632f'
#     user = api.getUserById(id)

#     sims = AI.recommend_weekly_tasks(user)

#     return sims

@app.post("/cluster")
def cluster():
    try:
        profile = request.json
        castedProfile  = RawProfile(user_id=profile['user_id'], scores=profile['scores'], preferences=profile['preferences'])
        cluster = int(AI.cluster_single_record(castedProfile))
        return jsonify({"cluster": cluster})
    except KeyError as e:
        return jsonify({"Error: Missing value": str(e)}), 400
    except ValueError as e:
        stringMsg = str(e)
        variable = re.search(r"\n(\w+)\n", stringMsg)
        match = re.search(r"type=(.*?),", stringMsg)
        expected_type = match.group(1)
        return jsonify({"Value Error": f'{variable.group(1)} should have {expected_type} value'}), 400


@app.post('/recommend_daily')
def recommend():
    try:
        recommendation_input = request.json
        casted = DailyTaskInput(
            user_id=recommendation_input['user_id'],
            scores=recommendation_input['scores'],
            preferences=recommendation_input['preferences'],
            cluster=recommendation_input['cluster'],
            work_end_time=recommendation_input['work_end_time'],
            sleep_time=recommendation_input['sleep_time']
        )
        recommended_tasks = AI.recommend_daily(casted)
        return jsonify(recommended_tasks), 200
    except KeyError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        stringMsg = str(e)
        variable = re.search(r"\n(\w+)\n", stringMsg)
        match = re.search(r"type=(.*?),", stringMsg)
        expected_type = match.group(1)
        return jsonify({"Value Error": f'{variable.group(1)} should have {expected_type} value'}), 400

@app.post('/recommend_weekly')
def recommend_weekly():
    try:
        recommendation_input = request.json
        casted = WeeklyRecommendationInput(user=recommendation_input['user'], work_end_time=recommendation_input['work_end_time'], work_start_time=recommendation_input['work_start_time'], sleep_time=recommendation_input['sleep_time'])
        recommended_tasks = AI.recommend_weekly_tasks(user=casted.user, work_end_time=casted.work_end_time, work_start_time=casted.work_start_time, sleep_time=casted.sleep_time)
        return jsonify(recommended_tasks), 200
    except KeyError as e:
        return jsonify({"Error: Missing value": str(e)}), 400
    except ValueError as e:
        stringMsg = str(e)
        variable = re.search(r"\n(\w+)\n", stringMsg)
        match = re.search(r"type=(.*?),", stringMsg)
        expected_type = match.group(1)
        return jsonify({"Value Error": f'{variable.group(1)} should have {expected_type} value'}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5120, debug=True)

