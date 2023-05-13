import sys
from pathlib import Path
import cv2
import tap
from tqdm import tqdm


class Arguments(tap.Tap):
    dataset: Path = Path("../dane/audio(wideo)/non_animal/")


FOURCC = cv2.VideoWriter_fourcc(*"mp4v")

XS = slice(860, 1207)
YS = slice(85, 730)

DIRS = [
    "A_non-contrast_1/A_non-contrast_1",
    "C_non-contrast_1/C_non-contrast_1",
    "D_non-contrast_1/D_non-contrast_1",
    "F_non-contrast_1/F_non-contrast_1",
]


def frames(capture):
    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            return

        yield frame


def process_file(dir: Path, file: Path):
    if not (file.is_file() and file.suffix.lower() == ".mp4"):
        return

    processed_dir = Path(str(dir) + "_processed")
    processed_dir.mkdir(exist_ok=True)
    processed = processed_dir / file.name

    cap = cv2.VideoCapture(str(file))
    # do not change last parameter
    out = cv2.VideoWriter(
        str(processed), FOURCC, 20, (XS.stop - XS.start, YS.stop - YS.start)
    )
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame in tqdm(frames(cap), total=frame_count, desc=file.name):
        therapy_frame = frame[YS, XS, :]
        out.write(therapy_frame)

    out.release()
    cap.release()


def main(dataset: Path):
    for dirname in DIRS:
        dir = dataset / dirname

        if not dir.exists():
            print(f"error: directory {dir} does not exist", file=sys.stderr)
            continue

        for file in dir.iterdir():
            process_file(dir, file)


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args.dataset)
