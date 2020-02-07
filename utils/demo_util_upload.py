from utils.edf2pickle_utils import read_edf, bandpower
from tensorflow.keras.models import load_model
from bokeh.io import save, output_file
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, RangeTool
from bokeh.plotting import figure
import numpy as np
import pickle


def check_filetype(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "edf"


def edf2psd(edf_name, subject_id):

    # define variables
    fdr = f"utils/raw_edf"
    ch_names = [
        "FP1-F7",
        "F7-T7",
        "T7-P7",
        "P7-O1",
        "FP1-F3",
        "F3-C3",
        "C3-P3",
        "P3-O1",
        "FP2-F4",
        "F4-C4",
        "C4-P4",
        "P4-O2",
        "FP2-F8",
        "F8-T8",
        "T8-P8",
        "P8-O2",
        "FZ-CZ",
        "CZ-PZ",
    ]

    # read edf file to numpy array
    f_path = f"{fdr}/{edf_name}"
    num_channel, sampling_freq, Nsamples, signal = read_edf(f_path, ch_names)

    # load scaler
    with open(f"utils/pickle_scaler/subject{subject_id}_scaler.pickle", "rb") as f:
        scaler = pickle.load(f)

    # segment signals
    f_duration = Nsamples // sampling_freq  # total number of seconds
    segments = [
        signal[:, i * sampling_freq : (i + 4) * sampling_freq].reshape(
            1, num_channel, -1
        )
        for i in range(f_duration - 4 + 1)
    ]
    segments = np.concatenate(segments, axis=0)
    # convert raw data into power
    bands = [[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 128]]
    bandpower_param = {"sf": 256, "bands": bands, "window_sec": 4}
    ps = np.apply_along_axis(
        bandpower, -1, segments, **bandpower_param
    )  # shape (sample, num_channel, band)
    ps = ps.reshape(-1, 90)
    ps = scaler.transform(ps)
    return f_duration, ps


def get_label(edf_name, f_duration):
    label = np.zeros(f_duration - 4 + 1)
    with open(f"utils/ref_dict.pickle", "rb") as f:
        ref_dict = pickle.load(f)
    events = ref_dict.get(edf_name, [])
    if len(events) != 0:
        idx = []
        for event in events:
            start, end = event
            idx += [i for i in range(start, end + 1, 1)]
        label[idx] = 1
    return label


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


def clf_predict(edf_name, subject_id):
    """
    Load pretrained model and predict
    """
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

    # get powerspectrum and file duration in sec
    f_duration, ps = edf2psd(edf_name, subject_id)
    # get doc's label
    label = get_label(edf_name, f_duration)
    # create Tx for LSTM
    Tx = Tx_dict.get(subject_id, 8)
    power_Tx, label_Tx = create_Tx(ps, label, Tx)
    # get feature indices
    rd_ch = ch_dict.get(subject_id)
    # rd_ch = [10, 11, 14, 15]
    rd_ch_idx = np.concatenate([np.arange((i - 1) * 5, i * 5) for i in rd_ch], axis=0)
    # load model and predict
    clf = load_model(f"utils/models/subject{subject_id}.h5")
    pred = clf.predict(power_Tx[:, :, rd_ch_idx])[:, 1]

    return pred, label_Tx


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


def make_plot(alarm, warning_msg, label_Tx=None):

    output_file("static/bokeh.html", title="Seizure Alarm")

    # define what to plot
    alarm_x = np.arange(alarm.shape[0])
    source = ColumnDataSource(data=dict(alarm_x=alarm_x, alarm=alarm))

    if label_Tx is not None:
        shift = label_Tx.shape[0] - alarm.shape[0]
        alarm = np.pad(alarm, (shift, 0), mode="constant", constant_values=0)
        label_Tx_x = np.arange(label_Tx.shape[0])
        source = ColumnDataSource(
            data=dict(
                alarm_x=label_Tx_x,
                alarm=alarm,
                label_Tx_x=label_Tx_x,
                label_Tx=label_Tx,
            )
        )

    # make an upper figure in the column object
    start = 1000
    end = 1060
    if alarm_x.shape[0] < 1060:
        start = 0
        end = 60
    p = figure(
        title=warning_msg,
        plot_height=400,
        plot_width=800,
        tools="xpan",
        toolbar_location=None,
        x_axis_location="above",
        background_fill_color="#efefef",
        x_range=(alarm_x[start], alarm_x[end]),
    )

    # draw alarm in upper figure
    p.line(
        "alarm_x",
        "alarm",
        legend_label="Alarm",
        line_width=3,
        color="red",
        source=source,
    )

    # make a lower figure in the column object
    select = figure(
        plot_height=200,
        plot_width=800,
        y_range=p.y_range,
        y_axis_type=None,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
    )

    # plot alarm in lower figure
    select.line("alarm_x", "alarm", color="red", source=source)

    if label_Tx is not None:
        # draw label in the upper figure
        p.line(
            "label_Tx_x",
            "label_Tx",
            legend_label="clinician label",
            line_width=3,
            color="orange",
            line_dash="4 4",
            source=source,
        )
        # draw label in the lower figure
        select.line(
            "label_Tx_x", "label_Tx", color="orange", line_dash="4 4", source=source
        )

    # cosmetic changes
    for panel in [p, select]:
        panel.yaxis.ticker = [0, 1]
        panel.yaxis.major_label_overrides = {0: "OFF", 1: "ON"}
    p.yaxis.axis_label = "Seizure"
    p.xaxis.axis_label = "Time (seconds)"
    p.title.text_font_size = "20pt"
    select.ygrid.grid_line_color = None

    # create a range tool
    range_tool = RangeTool(x_range=p.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    # add range tool to lower figure
    select.add_tools(range_tool)
    select.toolbar.active_multi = range_tool

    # save column object to html defined by output_file
    save(column(p, select))
