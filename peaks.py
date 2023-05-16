from pathlib import Path
import librosa
import numpy as np
import tap
import scipy.signal


class Config(tap.Tap):
    dataset: Path = Path("../dane/audio(wideo)")
    file: Path = Path("non_animal/D_non-contrast_1/D_non-contrast_1/REC69.WAV")
    # Path("animal/muscle_animal_1/muscle_animal_1/REC69.WAV")
    window: int = 1024
    split: float = 0.8
    order: int = 3
    lowcut: int = 512
    highcut: int = 2048
    percentile: int = 90
    hops: int = 128
    delta: float = 1.15


class PeakDetector(Config):
    def __init__(self) -> None:
        super().__init__()
        super().parse_args()
        self.y, self.sr = librosa.load(self.dataset / self.file, sr=None)
        self.y = self.normalize(self.slice(self.y, self.sr))

    def detect(self):
        y = self.filter(self.y)
        return self.peaks(y)

    @staticmethod
    def slice(y: np.ndarray, sr: float):
        duration = librosa.get_duration(y=y, sr=sr)
        return y[slice(*librosa.time_to_samples([1, duration - 1], sr=sr))]

    @staticmethod
    def normalize(y: np.ndarray):
        return librosa.util.normalize(y)

    def butter(self, lowcut: int, highcut: int, sr: float):
        return scipy.signal.butter(
            self.order, [lowcut / sr, highcut / sr], btype="band"
        )

    def filter(self, y: np.ndarray):
        return scipy.signal.lfilter(*self.butter(self.lowcut, self.highcut, self.sr), y)

    def peaks(self, y: np.ndarray):
        onset = librosa.onset.onset_strength(
            y=y,
            sr=self.sr,
            hop_length=self.hops,
            aggregate=lambda y, axis: np.percentile(y, self.percentile, axis=axis),
        )

        peaks = librosa.util.peak_pick(
            onset,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=self.delta,
            wait=10,
        )

        return librosa.frames_to_time(peaks, sr=self.sr, hop_length=self.hops)

    def windowed(self):
        step = int(self.window * self.split)
        return np.lib.stride_tricks.sliding_window_view(self.y, self.window)[::step]


def main(detector: PeakDetector):
    print(detector.detect())

    # peaks = [
    #     p
    #     for window in detector.windowed()
    #     for p in detector.peaks(detector.filter(window))
    # ]
    # print(peaks)


if __name__ == "__main__":
    detector = PeakDetector()
    main(detector)
