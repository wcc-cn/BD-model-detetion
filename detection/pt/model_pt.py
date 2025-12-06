from ultralytics import YOLO
from ultralytics.utils import LOGGER
import numpy as np
import cv2

class YoloModel:
    def __init__(self, model_path: str):
        LOGGER.setLevel("ERROR")
        self.model = YOLO(model_path, verbose=False)
        self.target_dict = self.model.names

    def predict(self, img: np.ndarray):
        ret = self.model.predict(img)
        detect_ret = []
        for x1, y1, x2, y2, conf, target_id in ret[0].boxes.data.cpu().numpy():
            detect_ret.append([int(x1), int(y1), int(x2), int(y2), float(f"{conf:.2f}"), self.target_dict[int(target_id)]])
        return detect_ret

    def draw_label_text(self, contexts, img: np.ndarray):
        if len(contexts) > 0:
            for item in contexts:
                text = f"{item[5]}: {item[4]}"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
                cv2.rectangle(img, (item[0], item[1] - text_h - 10), (item[0] + text_w, item[1]), (0, 0, 255),cv2.FILLED)
                cv2.putText(img, text, (item[0], item[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
