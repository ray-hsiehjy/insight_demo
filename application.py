from flask import Flask, render_template, request
from utils.demo_util import generate_fig
import matplotlib
import os

matplotlib.use("Agg")

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", title="Home")


@app.route("/result", methods=["POST"])
def result():
    subject_id = int(request.form.get("subject"))
    event_id = int(request.form.get("event_id"))
    threshold = request.form.get("threshold")
    sec_pre = 30
    sec_post = 30
    generate_fig(subject_id, event_id, sec_pre, sec_post, threshold)

    return render_template(
        "result.html",
        url_pred="static/img/prediction.png",
        url_avg=f"static/img/subject{subject_id}_avg.png",
        title="Result",
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
