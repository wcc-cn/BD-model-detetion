from detection.yolo_model import YoloModel
import cv2
import websockets
import asyncio
import json
import yaml
import os
from datetime import datetime
from threading import Thread
import queue
import time
import subprocess

def read_conf() -> dict:
    ret = {}
    if os.path.exists("./config.yaml"):
        with open("./config.yaml", 'r') as yaml_file:
            ret = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return ret

def detect_video(model: YoloModel, request_msg: dict, save_video_path: str):
    """ request
    {
        'cmd': 'detect_video',
        'data': {
            'sync_id': 'xxxxxxxxxxxxx',
            'video_path': 'xxxxxx'
        }
    }
    """
    """response
    {
        'cmd': 'detect_video_ack',
        'data': {
            'sync_id': 'xxxxxxxxxxxxx',
            'detect_target': ['aaa','bbb'],
            'fps': 25,
            'width': 1920,
            'height': 1080,
            'predict_video_path': 'xxxxxx',
            'origin_video_name':'xxxxxx'
        } 
    }
    """
    output_video = os.path.join(save_video_path, request_msg['data']['sync_id']+".mp4")
    temp_video = os.path.join(save_video_path, request_msg['data']['sync_id']+".avi")
    target_set = set()
    cap = cv2.VideoCapture(request_msg['data']['video_path'])

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        predict_ret = model.predict(frame)
        for item in predict_ret:
            target_set.add(item[5])
        model.draw_label_text(predict_ret, frame)
        out.write(frame)
    cap.release()
    out.release()
    cmd = (
        f"ffmpeg -y -i {temp_video} "
        f"-vcodec libx264 -pix_fmt yuv420p {output_video}"
    )
    subprocess.run(cmd, shell=True)
    os.remove(temp_video)

    response_msg = {
        'cmd': 'detect_video_ack',
        'data': {
            'sync_id': request_msg['data']['sync_id'],
            'detect_target': [item for item in target_set],
            'fps': fps,
            'width': width,
            'height': height,
            'predict_video_name': os.path.basename(output_video),
            'origin_video_name': os.path.basename(request_msg['data']['video_path'])
        }
    }
    return response_msg

def detect_main(request_queue: queue.Queue, response_queue: queue.Queue, conf_data: dict):
    print(f"[{datetime.now()}] model detect thread start ...")
    model = YoloModel(conf_data['server_conf']['model_path'])
    while True:
        try:
            request_msg = request_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.5)
            continue
        print(f"[{datetime.now()}] receive request msg : {request_msg}")
        if request_msg['cmd'] == 'detect_video':
            ret = detect_video(model, request_msg, conf_data['server_conf']['predict_video_save_path'])
            print(f"[{datetime.now()}] detect video finish, response : {ret}")
            response_queue.put(ret)

async def push_response(ws_client: websockets.ClientConnection, response_queue: queue.Queue):
    print(f"[{datetime.now()}] response push async start ...")
    while True:
        try:
            response_msg = response_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue
        await ws_client.send(json.dumps(response_msg))

async def ws_main(conf_data: dict, request_queue: queue.Queue, response_queue: queue.Queue):
    uri = conf_data['server_conf']['server_conf']
    async with websockets.connect(uri) as ws_client:
        await ws_client.send(json.dumps({'cmd':'connect', 'data':'keep live'}))
        push_response_task = asyncio.create_task(push_response(ws_client, response_queue))
        while True:
            msg = await ws_client.recv()
            if request_queue.full():
                request_queue.get_nowait()
            request_queue.put(json.loads(msg))
            await asyncio.sleep(0.1)

def main():
    conf_data = read_conf()
    if len(conf_data.keys()) < 0:
        print(f"[{datetime.now}] yaml config file is empty!")
        return
    request_queue = queue.Queue(maxsize=100)
    response_queue = queue.Queue(maxsize=100)
    detect_thread = Thread(target=detect_main, args=(request_queue, response_queue, conf_data))
    detect_thread.start()
    asyncio.run(ws_main(conf_data, request_queue, response_queue))
    detect_thread.join()

if __name__ == '__main__':
    main()