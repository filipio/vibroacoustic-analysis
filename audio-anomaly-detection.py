import os

import librosa
import numpy as np
from matplotlib import pyplot as plt
from streamad.model import (
    KNNDetector,
    LodaDetector,
    MadDetector,
    OCSVMDetector,
    RrcfDetector,
    SpotDetector,
    SRDetector,
)
from streamad.process import ZScoreCalibrator
from streamad.util import CustomDS, StreamGenerator, plot

SIGNAL_SAMPLING = 44_500
CLEAN_AUDIO_CUTOFF_INDEX = int(SIGNAL_SAMPLING / 2)
AUDIO_PEAK_INDEX_BOUNDARY = 300
AUDIO_PATH = (
    "../data/audio(wideo)/non_animal/D_non-contrast_1/D_non-contrast_1/REC70.WAV"
)
RESULTS_PATH = "./anomaly_detection_results"
SAVE_FILE_FORMAT = ".svg"


def load_audio_data(path):
    y, _ = librosa.load(path, sr=SIGNAL_SAMPLING)
    print("audio data loaded, len: ", len(y))
    return y


def plot_audio(signal):
    librosa.display.waveshow(signal, sr=SIGNAL_SAMPLING)
    plt.show()


def save_scores(path, dataset, scores):
    data, label, date, _ = dataset.data, dataset.label, dataset.date, dataset.features
    fig = plot(
        data=data, scores=scores, date=date, features=["original_audio"], label=label
    )

    fig.write_image(path)


def main():
    audio = load_audio_data(AUDIO_PATH)
    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    audio_clean = audio[CLEAN_AUDIO_CUTOFF_INDEX:-CLEAN_AUDIO_CUTOFF_INDEX]
    peak_index = np.argmax(audio_clean)
    original_peak_index = CLEAN_AUDIO_CUTOFF_INDEX + peak_index
    peak_audio = audio[
        original_peak_index
        - AUDIO_PEAK_INDEX_BOUNDARY : original_peak_index
        + AUDIO_PEAK_INDEX_BOUNDARY
    ]

    models = [
        SpotDetector(),
        KNNDetector(),
        LodaDetector(),
        MadDetector(),
        OCSVMDetector(),
        RrcfDetector(),
        SRDetector(),
    ]

    for model in models:
        model_name = model.__class__.__name__
        ds = CustomDS(peak_audio)
        stream = StreamGenerator(ds.data)
        print(model.__class__.__name__)
        calibrator = ZScoreCalibrator()
        scores = []
        scores_calibrated = []

        for x in stream.iter_item():
            score = model.fit_score(x)
            score_calibrated = calibrator.normalize(score)
            scores.append(score)
            scores_calibrated.append(score_calibrated)

        print(f"finished processing for model {model_name}")

        model_results_path = os.path.join(RESULTS_PATH, model_name + SAVE_FILE_FORMAT)
        calibrated_model_results_path = os.path.join(
            RESULTS_PATH, model_name + "_calibrated" + SAVE_FILE_FORMAT
        )

        save_scores(path=model_results_path, dataset=ds, scores=scores)
        save_scores(
            path=calibrated_model_results_path, dataset=ds, scores=scores_calibrated
        )


if __name__ == "__main__":
    main()
