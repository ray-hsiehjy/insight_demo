import glob
import os
import re
import pyedflib
import numpy as np
from scipy.signal import welch
from scipy.integrate import simps
import pickle


def read_edf(file_path: str, ch_names=None) -> (int, int, int, np.ndarray):
    """
    Load recoding data into a np.ndarray and extract some metadata
    Parameters:
    ----------
    file_path: str. path to a single edf file
    ch_names: 
        if a list. eg ["FP1-F7", "F7-T7"...]. Use standard 10-20 naming system. 
    Return:
    ----------
    num_channel, sampling_freq, Nsamples, signal
    """
    # Context manager "with" does not close EdfReader
    f = pyedflib.EdfReader(file_path)

    # get Nsamples
    Nsamples = f.getNSamples()
    assert len(set(Nsamples)) == 1
    Nsamples = Nsamples[0]

    # get sampling_freq
    sampling_freq = f.getSampleFrequencies()
    assert len(set(sampling_freq)) == 1
    sampling_freq = sampling_freq[0]

    # get signal labels
    sig_labels = f.getSignalLabels()
    # if no specific channels, return all in original order
    if ch_names == None:
        ch_names = sig_labels
    num_channel = len(ch_names)

    # create an all-zero 2d-array
    signal = np.zeros((num_channel, Nsamples))
    # get index of wanted channels
    ch_idx = [sig_labels.index(ch) for ch in ch_names]
    # read in specified channel one at a time
    for c, ch in enumerate(ch_idx):
        signal[c, :] = f.readSignal(ch)

    # explicitly close file
    f._close()
    del f

    return num_channel, sampling_freq, Nsamples, signal


def get_seizure_timestamps(path: str) -> dict:
    """
    Get seizure timestamps for an individual subject
    Parameters:
    ----------
    path: Path to subject's txt summary file
    Return:
    ----------
    ref_dict: Key:file_name. Value:np.ndarray, shape==(num of seizures, 2)
    """
    ref_dict = {}
    with open(path, "r") as f:
        paragraphs = f.read().split("\n\n")
        # remove extra lines and strip whitespaces
        paragraphs = [p.strip() for p in paragraphs if p != ""]
        for paragraph in paragraphs:
            # every paragraph has metadata info for one edf file
            if not paragraph.startswith("File Name:"):
                continue
            file_name = re.findall(r"chb\d{2}.*_\d{2}.*\.edf", paragraph)[0]
            # event_times is a list of timestamps of seizure events [start_1, end_1, start_2, end_2...]
            event_times = re.findall(r"\d+\sseconds", paragraph)
            # if no event in the recording file, event_times == empty list
            event_times = [int(timestamp.split(" ")[0]) for timestamp in event_times]
            # reshape into [[start_1, end_1], [start_2, end_2], ...]
            event_times = np.asarray(event_times).reshape(-1, 2)

            # assign seizure timestamps into reference dictionary
            ref_dict[file_name] = event_times

        return ref_dict


def bandpower(data, sf: float, bands: list, window_sec: float, relative=False) -> list:
    """
    Compute the average power of the signal x in a specific frequency band.
    Parameters:
    ----------
    data : 1d-array. Input signal in the time-domain.
    sf : Sampling frequency of the data.
    band : Lower and upper frequencies of the band of interest.
    window_sec : Length of each window in seconds.
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.
    Return:
    ----------
    bps : a list of absolute or relative power in decible.
    """

    # Define window length. nperseg: number of samples per segment
    nperseg = window_sec * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    bps = []
    for band in bands:
        low, high = band
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        # Integral approximation of the spectrum using Simpson's rule.
        bp = simps(psd[idx_band], dx=freq_res)
        if relative:
            bp /= simps(psd, dx=freq_res)
        # convert power unit to decible
        bp = 10 * np.log10(bp)
        bps.append(bp)
    return bps


def Edf_to_PickledArray(subject_id: int,):
    """
    Convert all edf file from one subject to concatenated numpy array and pickle the array
    
    Parameters:
    ----------
    subject_id: int
    Return:
    ----------
    create a subfolder "pickle_preictalXX" under main_folder and put three pickle files in it
        data X, shape (num_segment, num_ch, len(band_range))
        label y, shape (num_segment,): 0==interictal, 1==preictal, 2==ictal
    """

    subject_id = f"chb{str(subject_id).zfill(2)}"
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
    main_fdr = "../EEG_project/raw_data/"
    subject_fdr = os.path.join(main_fdr, f"{subject_id}")
    edf_lst = sorted(glob.glob(f"{subject_fdr}/*.edf"))

    with open(
        f"../EEG_project/pickle_ref_dict/{subject_id}_ref_dict.pickle", "rb"
    ) as f:
        ref_dict = pickle.load(f)

    X = []
    y = []
    for f in edf_lst:
        edf_name = os.path.split(f)[1]

        if edf_name not in ref_dict.keys():
            continue

        # read in single edf
        num_channel, sampling_freq, Nsamples, signal = read_edf(f, ch_names=ch_names)

        total_sec = Nsamples // sampling_freq  # total duration in int seconds
        # create samples sample duration = 4 sec; stride = 1 sec
        samples = [
            signal[:, i * 256 : (i + 4) * 256].reshape(1, num_channel, -1)
            for i in range(total_sec - 4 + 1)
        ]
        # concat all samples, shape (num_samples, num_channel, sampling_freq*sample_duration)
        samples = np.concatenate(samples, axis=0)

        # calculate power for every bandwidth, value unit in decible(db)
        bands = [[0.5, 4], [4, 8], [8, 12], [12, 30], [30, 128]]
        bandpower_param = {"sf": 256, "bands": bands, "window_sec": 4}
        # ps is a list of 5 numbers cooresponding 5 bandpowers
        ps = np.apply_along_axis(bandpower, -1, samples, **bandpower_param)
        ps = np.asarray(ps)

        # get event times
        events = ref_dict.get(edf_name)
        # create label array, preictal = 8 sec
        label = np.zeros(total_sec)  # label normal state as 0
        for event in events:
            start, end = event
            label[start - 8 : start] = 1  # extra preictal
            label[start:end] = 2  # ictal
        label = label[3:]

        assert label.shape[0] == ps.shape[0]  # number of samples
        assert ps.shape[1] == 18  # number of channels
        assert ps.shape[2] == 5  # number of bandpowers

        X.append(ps)
        y.append(label)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    with open(f"{subject_id}_PSD.pickle", "wb") as f:
        pickle.dump(X, f)
    with open(f"{subject_id}_label8.pickle", "wb") as f:
        pickle.dump(y, f)

    return f"{subject_id} done"
