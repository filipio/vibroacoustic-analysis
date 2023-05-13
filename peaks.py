from pathlib import Path
import librosa
import numpy as np
import tap


class Arguments(tap.Tap):
    dataset: Path = Path("../dane/audio(wideo)")
    file: Path = Path("animal/muscle_animal_1/muscle_animal_1/REC65.WAV")
    window_size: int = 1024


# onset = librosa.onset.onset_strength(
#     y=y, sr=sr, hop_length=512, aggregate=np.median
# )
# peaks = librosa.util.peak_pick(
#     onset, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10
# )
# print("Peaks detected at: ", peaks)


def windowed(data: np.ndarray, window: int, step: int = 1):
    """
    For given np.ndarray return generator that will yield array fragment of `window` length every `step`
    elements.
    """
    for i in range(window, len(data), step):
        yield data[i - window : i], i


def main(file: Path, window_size: int):
    y, sr = librosa.load(file)

    for window, i in windowed(y, window_size, int(window_size * 0.15)):
        split = int(window_size * 0.85)
        pre, post = window[:split], window[split:]
        ratio = np.median(pre) / np.percentile(post, 0.8)

        if ratio > 2:
            print(librosa.samples_to_time(i, sr=sr))


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args.dataset / args.file, args.window_size)
