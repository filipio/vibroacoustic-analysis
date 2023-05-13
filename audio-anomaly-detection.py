import librosa
import numpy as np
from matplotlib import pyplot as plt
from streamad.model import (
    KNNDetector,
    LodaDetector,
    MadDetector,
    OCSVMDetector,
    RrcfDetector,
    SArimaDetector,
    SpotDetector,
    SRDetector,
    ZScoreDetector,
    xStreamDetector,
)
from streamad.process import TDigestCalibrator, WeightEnsemble, ZScoreCalibrator
from streamad.util import CustomDS, StreamGenerator, plot

# ZScore fucked up

SIGNAL_SAMPLING = 44_500
CLEAN_AUDIO_CUTOFF_INDEX = int(SIGNAL_SAMPLING / 2)
AUDIO_PEAK_INDEX_BOUNDARY = 300
AUDIO_PATH = (
    "../data/audio(wideo)/non_animal/D_non-contrast_1/D_non-contrast_1/REC70.WAV"
)


def load_audio_data(path):
    y, _ = librosa.load(path, sr=SIGNAL_SAMPLING)
    print("audio data loaded, len: ", len(y))
    return y


def plot_audio(signal):
    librosa.display.waveshow(signal, sr=SIGNAL_SAMPLING)
    plt.show()


audio = load_audio_data(AUDIO_PATH)

audio_clean = audio[CLEAN_AUDIO_CUTOFF_INDEX:-CLEAN_AUDIO_CUTOFF_INDEX]
peak_index = np.argmax(audio_clean)
original_peak_index = CLEAN_AUDIO_CUTOFF_INDEX + peak_index
print(original_peak_index)
peak_audio = audio[
    original_peak_index
    - AUDIO_PEAK_INDEX_BOUNDARY : original_peak_index
    + AUDIO_PEAK_INDEX_BOUNDARY
]


ds = CustomDS(peak_audio)
stream = StreamGenerator(ds.data)
model = SpotDetector()
calibrator = ZScoreCalibrator()

scores = []
iter = 0

for x in stream.iter_item():
    score = model.fit_score(x)
    normalized_score = score
    # uncomment line below to use normalization calibrator
    # normalized_score = calibrator.normalize(score)
    scores.append(normalized_score)
    print(f"added score {iter + 1}")
    iter += 1

data, label, date, features = ds.data, ds.label, ds.date, ds.features
fig = plot(data=data, scores=scores, date=date, features=features, label=label)
fig.show()
