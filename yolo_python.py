from ctypes import *

import darknet
import os
import cv2
import numpy as np
import time
import re

hasGPU = True
lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)
#lib = CDLL("./darknet/libdarknet.so", RTLD_GLOBAL)


class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int),
                ("embeddings", POINTER(c_float)),
                ("embedding_size", c_int),
                ("sim", c_float),
                ("track_id", c_int)]

class DETNUMPAIR(Structure):
    _fields_ = [("num", c_int),
                ("dets", POINTER(DETECTION))]

class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

copy_image_from_bytes = lib.copy_image_from_bytes
copy_image_from_bytes.argtypes = [IMAGE,c_char_p]

predict = lib.network_predict_ptr
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
init_cpu = lib.init_cpu

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_batch_detections = lib.free_batch_detections
free_batch_detections.argtypes = [POINTER(DETNUMPAIR), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict_ptr
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

load_net_custom = lib.load_network_custom
load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
load_net_custom.restype = c_void_p

free_network_ptr = lib.free_network_ptr
free_network_ptr.argtypes = [c_void_p]
free_network_ptr.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_data = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

predict_image_letterbox = lib.network_predict_image_letterbox
predict_image_letterbox.argtypes = [c_void_p, IMAGE]
predict_image_letterbox.restype = POINTER(c_float)

network_predict_batch = lib.network_predict_batch
network_predict_batch.argtypes = [c_void_p, IMAGE, c_int, c_int, c_int,
                                   c_float, c_float, POINTER(c_int), c_int, c_int]
network_predict_batch.restype = POINTER(DETNUMPAIR)

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

class YOLO:
    net = None
    data = None

    def __init__(self,cfgPath,weightPath,dataPath,batch_size=1,gpus=0):
        set_gpu=(gpus)
        self.net = load_net_custom(cfgPath.encode('utf-8'),weightPath.encode('utf-8'), 0, 1) 
        self.data = load_data(dataPath.encode('utf-8'))
        self.data_names = []

        with open(dataPath) as dataFH:
            dataContents = dataFH.read()
            match = re.search("names *= *(.*)$", dataContents, re.IGNORECASE | re.MULTILINE)
            if match:
                result = match.group(1)
            else:
                result = None
            try:
                if os.path.exists(result):
                    with open(result) as namesFH:
                        namesList = namesFH.read().strip().split("\n")
                        self.data_names = [x.strip() for x in namesList]
            except TypeError:
                pass

    def detect(self, frame, thresh=.5, hier_thresh=.5, nms=.45):
        # arr1 = arr.transpose(2,0,1)
        # c = arr.shape[0]
        # h = arr.shape[1]
        # w = arr.shape[2]
        # arr1 = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        # mem = arr.ctypes.data_as(POINTER(c_float))
        # im = IMAGE(w,h,c,mem)
        image, arr = self.array_to_image(frame)

        image_width = image.w
        image_height = image.h
        num = c_int(0)
        pnum = pointer(num)
        predict_image(self.net, image)

        dets = get_network_boxes(self.net, image_width, image_height, thresh, hier_thresh, None, 0, pnum, 0)

        num = pnum[0]
        do_nms_sort(dets, num, self.data.classes, nms)

        res = []
        for j in range(num):
            for i in range(self.data.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.data_names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        del image, arr
        free_detections(dets, num)

        return res


    def array_to_image(self, arr):
        # arr = np.array.transpose(2,0,1)
        arr = arr.transpose(2,0,1)
        # arr = arr.swapaxes(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        mem = arr.ctypes.data_as(POINTER(c_float))
        image = IMAGE(w,h,c,mem)
        return image,arr

   



net = YOLO("./darknet/cfg/yolov4.cfg", "./darknet/yolov4.weights","./darknet/cfg/coco.data",1,1)
# url = "./darknet/data/Highway3.mp4"
# url = 'rtsp://admin:4ind331%23@192.168.0.242/profile2/media.smp'
# cap= cv2.VideoCapture(url)

# while True:
#     ret, frame = cap.read()
    
#     if ret:
        
#         result = net.detect(frame, 0.65, 0.65)
#         # print(result)
#         for detection in result:
            
#             label = detection[0]
#             confidence = detection[1]
#             labelText = label + ": " + str(np.rint(100 * confidence)) +"%"
#             x,y,w,h = detection[2]
#             # print(x,y,w,h)
#             xmin = int(round(x - (w / 2)))
#             xmax = int(round(x + (w / 2)))
#             ymin = int(round(y - (h / 2)))
#             ymax = int(round(y + (h / 2)))
#             # xmin, ymin, xmax, ymax = net.convertBounds(detection[2])

#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
#             cv2.putText(frame, labelText, (xmin,ymin-12), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,255), 1)

#         cv2.imshow('result', frame)

#         if cv2.waitKey(10) & 0xFF == 27:
#             break
#     else:
#         break
# cv2.destroyAllWindows()

# if __name__== '__main__':
#     YOLO("./cfg/yolov4-tiny.cfg", "./yolov4.weights", "data/obj.data",1)