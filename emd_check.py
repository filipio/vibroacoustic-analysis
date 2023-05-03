import emd
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter

SIGNAL_SAMPLING = 44_500


def load_audio_data(path):
    y, _ = librosa.load(path, sr=SIGNAL_SAMPLING)
    print("audio data loaded, len: ", len(y))
    return y


def filter_audio(signal, lowcut, highcut, fs, order=5):
    factor = 0.5 * fs
    low = lowcut / factor
    high = highcut / factor
    b, a = butter(order, [low, high], btype="band")
    return lfilter(b, a, signal)


def plot_audio(signal, sampling):
    librosa.display.waveshow(signal, sr=sampling)
    plt.show()


def save_imfs_to_files(imfs, sampling):
    for imf_index in range(imfs.shape[1]):
        imf = imfs[:, imf_index]
        plt.figure()
        plt.plot(imf[:])
        plt.savefig(f"imf_results/imf_{imf_index}")
        plt.close()


signal = load_audio_data(
    "../data/audio(wideo)/non_animal/C_non-contrast_1/C_non-contrast_1/REC70.wav"
)
signal = filter_audio(signal, lowcut=700, highcut=1500, fs=SIGNAL_SAMPLING)
plot_audio(signal, SIGNAL_SAMPLING)

imfs = emd.sift.sift(signal)
save_imfs_to_files(imfs, SIGNAL_SAMPLING)

IP, IF, IA = emd.spectra.frequency_transform(imfs, SIGNAL_SAMPLING, "hilbert")
freq_range = (0.1, 1000, 32)
hht_f, spec = emd.spectra.hilberthuang(
    IF, IA, freq_range, mode="amplitude", sum_time=False
)

signal_duration_in_seconds = int(librosa.get_duration(y=signal, sr=SIGNAL_SAMPLING))
time_vect = np.linspace(
    0, signal_duration_in_seconds, signal_duration_in_seconds * SIGNAL_SAMPLING
)

plt.figure(figsize=(8, 10))
plt.pcolormesh(
    time_vect[: len(time_vect)], hht_f, spec[:, : len(time_vect)], cmap="hot_r"
)
plt.title("Hilbert-Huang Transform")
plt.xlabel("Time (seconds)")
plt.ylabel("Frequency (Hz)")
plt.show()
