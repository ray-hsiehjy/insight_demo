import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils.eeg_util_pt2 import load_data


def event_indices(label_Tx, event_id, sec_pre, sec_post, preictal_length=8):
    """
    sec_pre: seconds to predict before event
    sec_post: seconds to predict after event
    return: the indices of start and end
    """
    stride = preictal_length // 4  # number of preictal seg per event
    start_index = np.where(label_Tx == 1)[0][::stride][event_id]

    start = start_index - ((sec_pre // 4) - 2)
    end = start_index + (sec_post // 4) + 2
    return start, end


def prediction(power_Tx, clf, start, end, subject_id):
    """
    power_Tx: input data
    clf: pretrained model
    start_index: output from event_start
    
    """
    if int(subject_id) == 12:
        ch_lst = [i for i in range(45, 55, 1)] + [i for i in range(65, 75, 1)]
    if int(subject_id) == 15:
        ch_lst = [i for i in range(10, 20, 1)] + [i for i in range(30, 40, 1)]
    if int(subject_id) == 24:
        ch_lst = [i for i in range(0, 10, 1)] + [i for i in range(10, 20, 1)]

    pred = clf.predict(power_Tx[start:end, :, ch_lst])
    return pred[:, 1]


def alarm_on(pred, threshold):
    """
    pred: pred_prob from prediction()
    threshold: either "L" or "H" from user input
    """
    if threshold == "L":
        trigger = 0.1
    else:
        trigger = 0.2
    delta = np.diff(pred)
    M = 5  # number of diff to cal cum_prob
    total_seg = pred.shape[0]
    alarm = [delta[i : i + M].sum() for i in range(total_seg - M + 1)]
    alarm = np.asarray(alarm) > trigger
    return alarm


def make_plot(pred, alarm, label_Tx, start, end, sec_pre):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    total_seg = end - start
    seg_shift = total_seg - alarm.shape[0]
    MD_label = label_Tx[start:end] > 1
    ax.scatter(np.arange(total_seg), pred, color="C0", label="Pred Prob", linewidth=3)
    ax.scatter(
        np.arange(total_seg),
        MD_label,
        color="C1",
        label="Clinician Label",
        linewidth=3,
        alpha=0.5,
    )
    ax.plot(
        np.arange(seg_shift, total_seg), alarm, color="r", label=f"Alarm",
    )

    ax.hlines(y=[0.5], xmin=0, xmax=total_seg, linestyles="--", color="k", alpha=0.4)
    # ax.vlines(x=np.arange(baseline, baseline+(preictal_length//4)), ymin=0, ymax=1, linestyles="--", color="k", alpha=0.5)
    # ax.vlines(x=total_seg-abnorm, ymin=0, ymax=1, linestyles="--", color="k")
    ax.set(
        xlabel="Time (seconds)",
        ylabel="Prob abnormal",
        xticks=np.arange(0, total_seg + 1, 5),
        xticklabels=np.arange(0, total_seg + 1, 5) * 4 - sec_pre,
    )
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig("static/img/prediction.png")


def generate_fig(subject_id, event_id, threshold, sec_pre, sec_post):
    """
    User inputs:
    subject_id, event_id, threshold, sec_pre, sec_post
    """
    train_ids = [int(subject_id)]
    event_id = int(event_id)
    sec_pre = int(sec_pre)
    sec_post = int(sec_post)
    # load data
    power_Tx, label_Tx = load_data(train_ids, preictal_length=8, Tx=3, local=True)
    # get event indices
    start, end = event_indices(label_Tx, event_id, sec_pre, sec_post, preictal_length=8)
    # load model
    clf_name = f"./utils/models/subject{subject_id}.h5"
    clf = load_model(clf_name)
    # make prediction
    pred = prediction(power_Tx, clf, start, end, int(subject_id))
    # calculation alarm
    alarm = alarm_on(pred, threshold=threshold)
    # make figure and save to insight_demo/img/prediction.png
    make_plot(pred, alarm, label_Tx, start, end, sec_pre)
