import tensorrt as trt
import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit

target_type = {
    0: "run",
    1: "stand",
    2: "lie",
    3: "sit"
}

class YoloTrtModel:
    def __init__(self, engine_path, conf_filter = 0.3):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.conf_filter = conf_filter
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self._prepare_io()

    def _prepare_io(self):
        self.bindings = []
        self.host_inputs = {}
        self.host_outputs = {}
        self.device_inputs = {}
        self.device_outputs = {}
        self.output_shapes = {}  # <<< 新增

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            size = int(np.prod(shape))
            nbytes = size * np.dtype(dtype).itemsize

            host_mem = cuda.pagelocked_empty(shape=(size,), dtype=dtype)
            device_mem = cuda.mem_alloc(nbytes)
            self.bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.host_inputs[name] = host_mem
                self.device_inputs[name] = device_mem
            else:
                self.host_outputs[name] = host_mem
                self.device_outputs[name] = device_mem
                self.output_shapes[name] = shape  # <<< 保存输出 shape

    def infer(self, input_numpy, input_name = "images"):
        np.copyto(self.host_inputs[input_name], input_numpy.ravel())
        cuda.memcpy_htod_async(self.device_inputs[input_name], self.host_inputs[input_name], self.stream)
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.context.set_tensor_address(name, self.bindings[i])
        self.context.execute_async_v3(self.stream.handle)
        for name, dev_ptr in self.device_outputs.items():
            cuda.memcpy_dtoh_async(self.host_outputs[name], dev_ptr, self.stream)
        self.stream.synchronize()
        return {k: v.copy() for k, v in self.host_outputs.items()}

    def preprocess_image(self, frame: np.ndarray, input_shape=(1, 3, 640, 640)):
        h0, w0 = frame.shape[:2]
        scale = min(input_shape[2] / h0, input_shape[3] / w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        img_resized = cv2.resize(frame, (new_w, new_h))
        top = (input_shape[2] - new_h) // 2
        bottom = input_shape[2] - new_h - top
        left = (input_shape[3] - new_w) // 2
        right = input_shape[3] - new_w - left
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, (144, 144, 144))
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_float, (2, 0, 1))
        img_input = np.expand_dims(img_chw, 0)
        return img_input, scale, top, left, (w0, h0)

    def postprocess_yolo(self, output, scale, top, left, ori_wh, conf_thres=0.3, iou_thres=0.5):
        output = output.reshape(1, output.shape[1], -1)  # 保证 3D
        output = output.transpose(0, 2, 1)[0]  # (num_boxes, 8)
        boxes, scores, classes = [], [], []

        for i in range(output.shape[0]):
            obj_item = output[i]
            cls_id = np.argmax(obj_item[4:])
            conf = np.max(obj_item[4:])
            if conf < conf_thres:
                continue
            x, y, w, h = obj_item[:4]
            # 去 padding + 缩放回原图
            x = (x - left) / scale
            y = (y - top) / scale
            w /= scale
            h /= scale
            boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
            scores.append(conf)
            classes.append(target_type[cls_id])

        ret = []
        # NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
            for i in indices:
                ret.append([int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3]), f'{scores[i]:.2f}',classes[i]])
        return ret

    def predict(self, frame: np.ndarray):
        img_tensor, scale, top, left, ori_wh = self.preprocess_image(frame)
        outputs = self.infer(img_tensor)
        # reshape 回模型输出 shape
        output_name = list(outputs.keys())[0]
        output_first = outputs[output_name].reshape(self.output_shapes[output_name])
        # 后处理
        ret = self.postprocess_yolo(output_first, scale, top, left, ori_wh, self.conf_filter)
        return ret

    def draw_label_text(self, contexts, img: np.ndarray):
        if len(contexts) > 0:
            for item in contexts:
                text = f"{item[5]}: {item[4]}"
                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (item[0], item[1]), (item[2], item[3]), (0, 0, 255), 2)
                cv2.rectangle(img, (item[0], item[1] - text_h - 10), (item[0] + text_w, item[1]), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, text, (item[0], item[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


if __name__ == "__main__":
    print("TensorRT 版本:", trt.__version__)
    model = YoloTrtModel("../../model/best.engine", 0.6)

    cap = cv2.VideoCapture("../../dataset/videos/run.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("break")
            break
        ret = model.predict(frame)
        model.draw_label_text(ret, frame)
        cv2.imshow("img", frame)
        cv2.waitKey(1)
