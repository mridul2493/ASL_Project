import os
import cv2
import mediapipe as mp
import numpy as np

SRC = "asl_dataset_digits"
OUT = "data"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
mp_drawing = mp.solutions.drawing_utils

for label in os.listdir(SRC):
    src_folder = os.path.join(SRC, label)
    dst_folder = os.path.join(OUT, label)
    os.makedirs(dst_folder, exist_ok=True)

    images = os.listdir(src_folder)[:300]  
    count = 0

    for img_name in images:
        img_path = os.path.join(src_folder, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            out_path = os.path.join(dst_folder, f"digit_{count}.txt")
            np.savetxt(out_path, np.array(data))
            count += 1

    print(f"✅ Saved {count} samples for digit {label}")