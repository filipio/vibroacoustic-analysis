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
from streamad.process import WeightEnsemble, ZScoreCalibrator
from streamad.util import CustomDS, StreamGenerator, plot

plt.style.use("./style.mplstyle")

SIGNAL_SAMPLING = 44_500
# about 0.5s is cut of from start/end of audio
CLEAN_AUDIO_CUTOFF_INDEX = int(SIGNAL_SAMPLING / 2)
AUDIO_PEAK_INDEX_BOUNDARY = 1000
AUDIO_PATH = (
    "../data/audio(wideo)/non_animal/D_non-contrast_1/D_non-contrast_1/REC69.WAV"
)
RESULTS_PATH = os.path.join(
    "./anomaly_detection_results", AUDIO_PATH.split("/")[-1][:-4]
)
SAVE_FILE_FORMAT = ".pdf"


def load_audio_data():
    y, _ = librosa.load(AUDIO_PATH, sr=SIGNAL_SAMPLING)
    print("audio data loaded, len: ", len(y))
    return y


def extract_audio_peak_part(audio):
    audio_clean = audio[CLEAN_AUDIO_CUTOFF_INDEX:-CLEAN_AUDIO_CUTOFF_INDEX]
    peak_index = np.argmax(audio_clean)
    original_peak_index = CLEAN_AUDIO_CUTOFF_INDEX + peak_index
    result = audio[
        original_peak_index
        - AUDIO_PEAK_INDEX_BOUNDARY : original_peak_index
        + AUDIO_PEAK_INDEX_BOUNDARY
    ]
    return result


def plot_wave(signal):
    librosa.display.waveshow(signal, sr=SIGNAL_SAMPLING)
    plt.show()


def plot_audio(signal):
    plt.figure()
    plt.plot(signal, label="VAS")
    plt.ylabel("value")
    plt.legend()
    plt.show()


def save_anomaly_scores(path, dataset, scores):
    data, label, date, _ = dataset.data, dataset.label, dataset.date, dataset.features
    fig = plot(
        data=data, scores=scores, date=date, features=["original_audio"], label=label
    )

    fig.write_image(path)


def with_ensemble(audio, ensembles):
    all_results = []
    ds = CustomDS(audio)
    stream = StreamGenerator(ds.data)
    for x in stream.iter_item():
        results = []
        for detectors, calibrators, ensemble_calculator, _ in ensembles:
            assert len(detectors) == len(calibrators)
            calibrated_scores = []

            for i in range(len(detectors)):
                detector, calibrator = detectors[i], calibrators[i]
                score = detector.fit_score(x)
                calibrated_score = calibrator.normalize(score)
                calibrated_scores.append(calibrated_score)

            results.append(ensemble_calculator.ensemble(calibrated_scores))

        all_results.append(results)
    return np.array(all_results)


def main():

    if not os.path.exists(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    audio = load_audio_data()
    peak_audio = extract_audio_peak_part(audio)
    ds = CustomDS(peak_audio)
    # plot_audio(peak_audio)
    ensembles = [
        (
            (RrcfDetector(), MadDetector()),
            (ZScoreCalibrator(), ZScoreCalibrator()),
            WeightEnsemble(ensemble_weights=[0.4, 0.6]),
            "rrcf+mad_04_06",
        ),
        (
            (RrcfDetector(), MadDetector()),
            (ZScoreCalibrator(), ZScoreCalibrator()),
            WeightEnsemble(ensemble_weights=[0.2, 0.8]),
            "rrcf+mad_02_08",
        ),
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

    ensembled_results = with_ensemble(peak_audio, ensembles)
    for i in range(len(ensembles)):
        ensemble_scores = ensembled_results[:, i]
        model_name = ensembles[i][-1]
        ensemble_path = os.path.join(RESULTS_PATH, model_name + SAVE_FILE_FORMAT)
        save_anomaly_scores(path=ensemble_path, dataset=ds, scores=ensemble_scores)

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

        save_anomaly_scores(path=model_results_path, dataset=ds, scores=scores)
        save_anomaly_scores(
            path=calibrated_model_results_path, dataset=ds, scores=scores_calibrated
        )


if __name__ == "__main__":
    main()
