from edf2pickle_utils import read_edf, bandpower
import numpy as np
import pickle
import time


def edf2psd(f,):
    pass


t1 = time.perf_counter()

# define variables
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
subject_id = 12
f_name = "chb12_06.edf"
fdr = f"/Users/rayhsieh/Desktop/EEG_project/raw_data/chb{subject_id}"

# read edf file to numpy array
f_path = f"{fdr}/{f_name}"
num_channel, sampling_freq, Nsamples, signal = read_edf(f_path, ch_names)

# load scaler
with open(f"./pickle_scaler/subject{subject_id}_scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# segment signals
f_duration = Nsamples // sampling_freq  # total number of seconds
segments = [
    signal[:, i * sampling_freq : (i + 4) * sampling_freq].reshape(1, num_channel, -1)
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

t2 = time.perf_counter()

print(t2 - t1)

print(ps.shape)
