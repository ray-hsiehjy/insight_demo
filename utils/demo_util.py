import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(subject_id: int):
    """
    subject_id: int 
    main_fdr: str
    clean: boolean, replace power < 0 with 0
    """
    main_fdr = "./utils"
    subject = f"chb{str(subject_id).zfill(2)}"

    psd_pickle = f"{main_fdr}/pickle_psd/{subject}_PSD.pickle"
    label_pickle = f"{main_fdr}/pickle_labels/{subject}_label8.pickle"

    with open(psd_pickle, "rb") as f:
        power = pickle.load(f)  # power.shape = (sample, num_ch, num_bands)
        power = power.reshape(-1, 90)
    with open(label_pickle, "rb") as f:
        label = pickle.load(f)

    power = np.where(power >= 0, power, 0)

    return power, label


def create_Tx(X, y, Tx):
    """
    Parameter:
        X: power to be binned
        y: label to be binned
        Tx: time-steps for lstm. in seconds
    Return:
        binned_X: shape (num_sample, Tx, 90)
        binned_y
    """
    # num_ch = 18, num_bands=5, feature dim = 18*5
    binned_X = [
        X[i : i + Tx, :].reshape(-1, Tx, 90) for i in range(X.shape[0] - Tx + 1)
    ]
    binned_X = np.concatenate(binned_X, axis=0)
    binned_y = [y[i : i + Tx].max() for i in range(y.shape[0] - Tx + 1)]
    binned_y = np.asarray(binned_y)
    assert binned_X.shape[0] == binned_y.shape[0]

    return binned_X, binned_y


def get_event_indices(subject_id, label_Tx, pre_sec, post_sec):

    starts = np.where(label_Tx == 1)[0][7::8] + 1  # first label==2 of every event
    event_indices = [np.arange(start - pre_sec, start + post_sec) for start in starts]
    event_indices = np.concatenate(event_indices, axis=0).reshape(
        -1, pre_sec + post_sec
    )
    return event_indices


def alarm_on(pred, threshold):
    """
    pred: pred_prob from prediction()
    threshold: either "L" or "H" from user input
    """
    if threshold == "L":
        trigger = 0.1
    elif threshold == "M":
        trigger = 0.25
    else:
        trigger = 0.4
    delta = np.diff(pred)
    M = 5  # number of diff to cal cum_prob
    total_seg = pred.shape[0]
    alarm = [delta[i : i + M].sum() for i in range(total_seg - M + 1)]
    alarm = np.asarray(alarm) > trigger
    return alarm


def make_plot(
    pred, alarm, label_Tx, event_indices_single, pre_sec, post_sec, threshold
):
    sns.set()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    total_seg = pre_sec + post_sec
    seg_shift = total_seg - alarm.shape[0]
    MD_label = label_Tx[event_indices_single] > 1

    # plot predicted prob
    sns.lineplot(
        np.arange(total_seg),
        pred,
        color="C0",
        label="Pred Prob",
        linewidth=3,
        linestyle="--",
        ax=ax,
    )
    # plot doc label
    sns.lineplot(
        np.arange(total_seg),
        MD_label,
        color="C1",
        label="Clinician Label",
        linewidth=3,
        linestyle="--",
        ax=ax,
    )
    # plot alarm
    sns.lineplot(
        np.arange(seg_shift, total_seg),
        alarm,
        color="r",
        label=f"Alarm={threshold}",
        linewidth=5,
        ax=ax,
    )
    # plot ref line
    ax.hlines(y=[0.5], xmin=0, xmax=total_seg, linestyles="--", color="k", alpha=0.4)
    ax.set_xlabel("Time (seconds)", size=16)
    ax.set_ylabel("Prob abnormal", size=16)
    ax.legend(loc="upper left", fontsize="large")
    # save figure for individual event
    fig.savefig(f"static/img/prediction.png")


def generate_fig(subject_id, event_id, pre_sec, post_sec, threshold):

    Tx_dict = {16: 16, 18: 12}
    ch_dict = {
        1: [9, 10, 13, 14],
        3: [1, 5, 9, 13],
        5: [11, 12, 15, 16],
        6: [5, 6, 9, 10],
        8: [5, 6, 9, 10],
        10: [2, 3, 4, 8],
        12: [10, 11, 14, 15],
        14: [3, 4, 15, 16],
        15: [3, 4, 7, 8],
        20: [2, 3, 6, 7],
        23: [1, 2, 5, 6],
        24: [9, 10, 13, 14],
        # not learning well, abnorm 0.65 < F1 <0.75
        16: [i for i in range(1, 19, 1)],
        18: [i for i in range(1, 19, 1)],
        # not learning
        13: [i for i in range(1, 19, 1)],
    }

    # load data
    power, label = load_data(subject_id)

    # tranform power
    with open(f"utils/pickle_scaler/subject{subject_id}_scaler.pickle", "rb") as f:
        scaler = pickle.load(f)
    power = scaler.transform(power)

    # get Tx
    Tx = Tx_dict.get(subject_id, 8)  # define Tx

    # Create_Tx
    power_Tx, label_Tx = create_Tx(power, label, Tx)

    # get event indices and select subset of power_Tx
    event_indices = get_event_indices(subject_id, label_Tx, pre_sec, post_sec)
    event_indices_single = event_indices[event_id]

    # get feature indices
    rd_ch = ch_dict.get(subject_id)
    # rd_ch = [10, 11, 14, 15]
    rd_ch_idx = np.concatenate([np.arange((i - 1) * 5, i * 5) for i in rd_ch], axis=0)

    # select subset of power_Tx
    power_Tx = power_Tx[:, :, rd_ch_idx]

    # load model and predict
    clf = load_model(f"utils/models/subject{subject_id}.h5")
    pred = clf.predict(power_Tx[event_indices_single, :, :])[:, 1]
    # pred = pred.reshape(-1, pre_sec + post_sec)

    # calculate alarm
    alarm = alarm_on(pred, threshold)
    # make and save plot

    make_plot(pred, alarm, label_Tx, event_indices_single, pre_sec, post_sec, threshold)

