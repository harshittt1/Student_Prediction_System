from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


perf_model = pickle.load(open('model/performance_model.pkl', 'rb'))  # Logistic
place_model = pickle.load(open('model/placement_model.pkl', 'rb'))   # Random Forest


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form
        action = request.form.get('action')
        
        advice = []
        perf_result = None
        place_result = None

        # ---------------- PERFORMANCE INPUTS ----------------
        hours = request.form.get('hours')
        attendance = request.form.get('attendance')
        sleep = request.form.get('sleep')
        previous = request.form.get('previous')
        internet = request.form.get('internet')

        # Check if performance inputs are filled
        if hours and attendance and sleep and previous and internet:

            hours = float(hours)
            attendance = float(attendance)
            sleep = float(sleep)
            previous = float(previous)
            internet = 1 if internet == 'yes' else 0

            perf_input = np.array([[hours, attendance, sleep, previous, internet]])
            perf_pred = perf_model.predict(perf_input)[0]

            perf_result = "Pass" if perf_pred == 1 else "Fail"

            # Advice (only if used)
            if hours < 5:
                advice.append("Increase study hours")
            if attendance < 75:
                advice.append("Improve attendance")

        # ---------------- PLACEMENT INPUTS ----------------
        cgpa = request.form.get('cgpa')
        internship = request.form.get('internship')
        communication = request.form.get('communication')
        projects = request.form.get('projects')
        extra = request.form.get('extra')

        # Check if placement inputs are filled
        if cgpa and internship and communication and projects and extra:

            cgpa = float(cgpa)
            internship = 1 if internship == 'yes' else 0
            communication = float(communication)
            projects = float(projects)
            extra = float(extra)

            place_input = np.array([[cgpa, internship, communication, projects, extra]])
            place_pred = place_model.predict(place_input)[0]

            place_result = "Placed" if place_pred == 1 else "Not Placed"

            # Advice
            if cgpa < 7:
                advice.append("Improve CGPA")
            if projects < 2:
                advice.append("Work on more projects")

        # ---------------- NO INPUT CASE ----------------
        if perf_result is None and place_result is None:
            return "Please fill at least one section"

        return render_template('index.html',
                               perf_result=perf_result,
                               place_result=place_result,
                               advice=advice,
                               form_data=request.form)


    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)