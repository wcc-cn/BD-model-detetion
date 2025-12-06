import onnxruntime as ort
import numpy as np
import cv2

target_type = {
    0: "run",
    1: "stand",
    2: "lie",
    3: "sit"
}

class YoloOnnxModel:
    def __init__(self, model_path, conf_filter: float = 0.3):
        self.model_path = model_path
        self.conf_filter = conf_filter
        self.session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name

    def pretreatment(self, frame: np.ndarray):
        origin_hw = frame.shape[:2]
        scale = min((self.input_shape[2] / origin_hw[0]), (self.input_shape[3] / origin_hw[1]))
        new_wh = (int(origin_hw[1] * scale), int(origin_hw[0] * scale))
        resize_frame = cv2.resize(frame, new_wh, interpolation = cv2.INTER_LINEAR)
        top_border = int((self.input_shape[2]  - new_wh[1])/2)
        bottom_border = int((self.input_shape[2]  - new_wh[1])/2)
        left_border = int((self.input_shape[3] - new_wh[0])/2)
        right_border = int((self.input_shape[3] - new_wh[0])/2)
        new_frame = cv2.copyMakeBorder(resize_frame, top_border, bottom_border, left_border, right_border,cv2.BORDER_CONSTANT, (144,144,144))
        new_frame = np.expand_dims(new_frame/255, axis=0)
        frame_pre = np.transpose(new_frame, (0,3,1,2)).astype(np.float32)
        return frame_pre, scale, top_border, left_border

    def post_process(self, predict_output, scale, top_border, left_border):
        output = np.transpose(np.squeeze(predict_output))
        boxes, scores, classes = [], [], []
        for i in range(output.shape[0]):
            if np.max(output[i][4:]) < self.conf_filter:
                continue
            x, y, w, h = output[i][:4]
            w /= scale
            h /= scale
            x = (x-left_border)/scale-w/2
            y = (y-top_border)/scale-h/2
            boxes.append([x, y, w, h])
            scores.append(max(output[i][4:]))
            classes.append(target_type[np.argmax(output[i][4:])])
        # 非极大值抑制 过滤重合度高的框
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_filter, 0.5)
        ret = []
        for i in indices:
            x, y, w, h = boxes[i]
            item = [int(x), int(y), int(x+w), int(y+h), f"{scores[i]:.2f}", classes[i]]
            ret.append(item)
        return ret

    def predict(self, frame: np.ndarray):
        # 1.图像预处理
        frame_pre, scale, top_border, left_border = self.pretreatment(frame)
        # 2.模型推理
        output = self.session.run(None, {self.input_name: frame_pre})
        # 3.后处理
        predict_ret = self.post_process(output, scale, top_border, left_border)
        return predict_ret

    def draw_label_text(self, contexts, img: np.ndarray):
        if len(contexts) > 0:
            for item in contexts:
                text = f"{item[5]}: {item[4]}"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
                cv2.rectangle(img, (item[0], item[1]-text_h-10), (item[0]+text_w, item[1]), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, text, (item[0], item[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 0, 0), 1)

if __name__ == "__main__":
    img = cv2.imread("../../dataset/imgs/123.png")
    model = YoloOnnxModel("D:/code/project/python/BD-model-detetion/model/best.onnx")
    predict_ret = model.predict(img)
    model.draw_label_text(predict_ret, img)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    print(predict_ret)