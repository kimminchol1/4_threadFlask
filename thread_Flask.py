from os import name
import threading
import cv2
import time
import numpy as np
from collections import deque
from flask import Flask,Response,render_template
from darknet.yolo_python import net


app = Flask(__name__)
disp_frame = {}
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cctv')
def cctv():
    return Response(getFrames('cctv'), mimetype = 'multipart/x-mixed-replace; boundary=frame')

@app.route('/webcam')
def webcam():
    # return render_template('index.html')
    return Response(getFrames('webcam'), mimetype = 'multipart/x-mixed-replace; boundary=frame')





def getFrames(name):
    global disp_frame

    while True:
        if disp_frame[name] is not None:
            ret, jpeg = cv2.imencode('.jpg', disp_frame[name], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            
            bframe = jpeg.tobytes()
            if bframe is not None:
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + bframe + b'\r\n\r\n')
        else :
             yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n\r\n\r\n')

class threadA(threading.Thread):
    def __init__(self, url=None, name=None):
        threading.Thread.__init__(self)
        
        self.cap = None
        self.name = name
        self.frame = None
        self.frame_count = 0
        self.results = []

        if url is not None:
            self.cap = cv2.VideoCapture(url)
            self.cap.set(3, 640)
            self.cap.set(4, 480)
            


    def run(self):
        while True:
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, dsize=(640,480), interpolation=cv2.INTER_AREA)
            if ret:
                self.frame = frame
                self.frame_count += 1
                if (self.frame_count%3 == 0) & (len(th_detect.Q) < 30):
                    th_detect.Q.append([self, frame])


class threadB(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.Q = deque()

    def run(self):
        while True:
            if len(self.Q) > 0 :
                th_A, frame = self.Q.popleft()
                #~~detect~~
                results = net.detect(frame, 0.5, 0.5)
                th_A.results = results
            time.sleep(0.0001)
        


class threadC(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global disp_frame
        
        while True:
            prev_time = time.time()
            for th_A in thread_list:
                if th_A.frame is not None:
                    frame = th_A.frame.copy()
                    
                    for detection in th_A.results:
                        label = detection[0]
                        confidence = detection[1]
                        x,y,w,h = detection[2]
                        xmin = int(round(x - (w / 2)))
                        xmax = int(round(x + (w / 2)))
                        ymin = int(round(y - (h / 2)))
                        ymax = int(round(y + (h / 2)))

                        labelText = label + ": " + str(np.rint(100 * confidence)) +"%"

                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
                        cv2.putText(frame, labelText, (xmin,ymin-12), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,255), 1)
                        
                    
                    cv2.imshow(th_A.name, frame)
                    disp_frame[th_A.name] = frame
                    # print(1/(time.time()-prev_time))
                    # prev_time = time.time()

            if cv2.waitKey(1) & 0xff == 27:
                cv2.destroyAllWindows()
                exit()

            time.sleep(1/30)


if __name__ == '__main__':

    cctv = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
    th_detect = threadB()
    th_detect.start()

    thread_list = []
    thread_list.append(threadA(url=1, name='webcam'))
    thread_list.append(threadA(url=cctv, name='cctv'))
    for thr in thread_list:
        thr.start()

    th_output = threadC()
    th_output.start()
    
    app.run(host='127.0.0.1', debug = False, port=5000)
