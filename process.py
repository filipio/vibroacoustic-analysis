import os

import cv2

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
x_start = 860
x_end = 1207
y_start = 85
y_end = 730

base_dir = "../data/audio(wideo)/non_animal/"

dirs = [
    "A_non-contrast_1/A_non-contrast_1",
    "C_non-contrast_1/C_non-contrast_1",
    "D_non-contrast_1/D_non-contrast_1",
    "F_non-contrast_1/F_non-contrast_1",
]

for data_dir in dirs:
    dir = os.path.join(base_dir, data_dir)
    for file_name in os.listdir(dir):
        if file_name[-3:].lower() == "mp4":
            print("found mp4 : ", file_name)
            mp4_path = os.path.join(dir, file_name)
            processed_dir = dir + "_processedd"
            if not os.path.exists(processed_dir):
                os.mkdir(processed_dir)
            processed_mp4_path = os.path.join(processed_dir, file_name)

            cap = cv2.VideoCapture(mp4_path)
            # do not change last parameter
            out = cv2.VideoWriter(processed_mp4_path, fourcc, 20, (347, 645))

            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    print("finished processing file", file_name)
                    break
                therapy_frame = frame[y_start:y_end, x_start:x_end, :]
                out.write(therapy_frame)

            out.release()
            cap.release()
