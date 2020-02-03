from flask import Flask, render_template, request, redirect, send_from_directory
from utils.demo_util_upload import clf_predict, alarm_on, make_plot
import pickle
import numpy as np
import time

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/")
@app.route("/home", methods=["GET", "POST"])
def home():
    if request.method == "POST":
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
            warning_msg = "Seizure Detected!"
        else:
            warning_msg = "Seizure NOT Detected!"

        # call for bokeh plot as static file
        make_plot(alarm, warning_msg, label_Tx=label_Tx)

        return send_from_directory("static", "bokeh.html")
    return render_template("home.html", title="Home")


@app.route("/upload")
def upload():
    return render_template("upload.html")


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")

