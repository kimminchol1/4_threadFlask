
import cv2
import time
from ctypes import *
import numpy as np
import os
import threading
from darknet.yolo_python import net
from collections import deque
from flask import Flask,Response

app = Flask(__name__)
disp_frame = None
@app.route('/')
def server_stream():
    # return render_template('index.html')
    return Response(getFrames(), mimetype = 'multipart/x-mixed-replace; boundary=frame')


def getFrames():
    global disp_frame

    while True:
        if disp_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', disp_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            
            bframe = jpeg.tobytes()
            if bframe is not None:
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + bframe + b'\r\n\r\n')
        else :
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n\r\n\r\n')


class First(threading.Thread):
    def __init__(self, url=None, name=None, thread=None):

        threading.Thread.__init__(self)

        self.cap = None
        self.name = name
        self.second_thread = thread
        if url is not None:
            
            self.cap = cv2.VideoCapture(url)
            self.cap.set(3, 640)
            self.cap.set(4, 480)
        # self.disp_frame = None
        

    def run(self):
        while True:
            
            ret, frame = self.cap.read()
            if ret :
                if len(self.second_thread.first_deque) < 20:
                    self.second_thread.first_deque.append([frame,self.name])


class Second(threading.Thread):
    
    def __init__(self):
        
        threading.Thread.__init__(self)
        self.first_deque = deque()
        # self.first_thread = thread
        self.results = []  

    def run(self):
        prev_time = time.time()
        
        global disp_frame
        while True:
            if len(self.first_deque) > 0:
                second_frame = self.first_deque.popleft()
                second_frame[0] = cv2.resize(second_frame[0], dsize=(640, 480), interpolation=cv2.INTER_AREA)
                results = net.detect(second_frame[0], 0.5, 0.5)
                for detection in results:
                    label = detection[0]
                    confidence = detection[1]
                    x,y,w,h = detection[2]
                    xmin = int(round(x - (w / 2)))
                    xmax = int(round(x + (w / 2)))
                    ymin = int(round(y - (h / 2)))
                    ymax = int(round(y + (h / 2)))

                    labelText = label + ": " + str(np.rint(100 * confidence)) +"%"

                    cv2.rectangle(second_frame[0], (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                    cv2.putText(second_frame[0], labelText, (xmin,ymin-12), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,255), 1)
                    disp_frame = second_frame[0]
                cv2.imshow(second_frame[1], second_frame[0])
                cur_time = time.time()
                if cur_time - prev_time > 0:
                    fps = 1/(cur_time - prev_time)
                    print(fps) #평균 29~31. 최저 12 최고 60
                    prev_time = cur_time
                
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cv2.destroyAllWindows()

# class Third(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#         self.third_deque = {}
#         # self.r

# Third = Third()

if __name__ == '__main__':

    thread_list = []
    thread_list.append(Second())
    # thread_list.append(First(url=0, name= 'webcam', thread=thread_list[0]))
    thread_list.append(First(url='rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp', name='cctv', thread=thread_list[0]))


    for thr in thread_list:
        thr.start()
    app.run(host='0.0.0.0', debug = False, port=5000)
# while True:
#     if First.disp_frame is not None:
        
#         cv2.imshow('copy', First.disp_frame)
        
        
#         if cv2.waitKey(10) & 0xFF == 27:
#             break