from flask import Flask, render_template, request, redirect
from utils.demo_util_upload import clf_predict, alarm_on, make_plot
import pickle
import numpy as np

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")


@app.route("/result", methods=["POST"])
def result():

    # get var from submitted form
    subject_id = int(request.form.get("subject"))
    edf_name = request.form.get("edf_name")
    threshold = request.form.get("threshold")

    # predict prob
    pred, label_Tx = clf_predict(edf_name, subject_id)

    # cal alarm on/off
    alarm = alarm_on(pred, threshold)

    # difine warning_msg
    detected = alarm.sum()
    if detected >= 1:
        warning_msg = "Seizure Detected!!"
    else:
        warning_msg = "Seizure NOT Detected!!"

    # call for bokeh plot
    make_plot(alarm, label_Tx, warning_msg)

    return render_template("result.html", warning=warning_msg, title="Result",)


@app.route("/upload")
def upload():
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")

