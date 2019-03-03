import numpy as np
import cv2
import copy
import time
import os
import rospy
import roslib
import tensorflow as tf
from scipy.signal import find_peaks
from keras.models import load_model
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from model import *

DIM = (402, 256)
K = np.array([[140.53356603691236, 0.0, 205.88099735997747], [0.0, 107.55850644017589, 129.01739742335653], [0.0, 0.0, 1.0]])
D = np.array([[0.1590449991308693], [-0.03899384165379903], [0.058623713311999205], [-0.030283257738173665]])

def undistort(img):
    h,w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def white_yello(img):
    cond1 = img[:,:,1] < 50
    cond2 = img[:,:,2] < 50
    cond = cond1 | cond2
    img[cond] = 0
    img[~cond] = 255
    return img

def find_peak(img, pos, width):
    peak = {}
    h, w, c = img.shape

    for i in reversed(range(16)):
        w1 = pos[i] - width / 2
        w2 = pos[i] + width / 2
        
        if pos[i] - width / 2 < 0:
            w1 = 0

        if pos[i] + width / 2 > w:
            w2 = w

        # cv2.rectangle(img, (w1, (i + 1) * 20), (w2, (i + 2) * 20), (0, 255, 0), 1)
        roi = img[(i + 1) * 20:(i + 2) * 20,w1:w2,:]
        mask = np.logical_and(roi[:,:,0] != 0, roi[:,:,1] != 0, roi[:,:,2] != 0)
        x = np.sum(mask, axis=0)
        peaks, _ = find_peaks(x, height=10, distance=10)
        peak[i] = []

        for j in peaks:
            peak[i].append(int(j) + w1)
            # cv2.circle(img, (int(j) + w1, (i + 1) * 20 + 10), 3, (255, 0, 0), -1)
    
    return img, peak

# h x w = 360 x 224
def image_callback(msg):
    global bridge
    global model
    global graph
    global left_pos
    global right_pos

    frame = bridge.imgmsg_to_cv2(msg, "bgr8")
    top = copy.copy(frame[60:420,:224,:])
    top[93:267, 61:150] = 0
    h, w, c = top.shape
    top = np.reshape(top, (1, h, w, c))
    top = np.array(top, dtype=np.float64)
    top /= 255

    with graph.as_default():
        prediction =  model.predict(top)[0]*255

    prediction = np.array(prediction, dtype=np.uint8)

    # front = copy.copy(frame[60:316,238:800,:])
    # front = undistort(front)
    # cv2.imshow('front', front)
    
    prediction = white_yello(prediction)

    left = prediction[:,:100,:]
    right = prediction[:,124:,:]

    width = 60

    left, left_peak = find_peak(left, left_pos, width)
    right, right_peak = find_peak(right, right_pos, width)

    left_pt_x = []
    left_pt_y = []
    right_pt_x = []
    right_pt_y = []
    next_left_pos = []
    next_right_pos = []

    f1 = lambda y, a, b: a*y + b
    f2 = lambda y, a, b, c: a*y**2 + b*y + c
    f3 = lambda y, a, b, c, d: a*y**3 + b*y**2 + c*y + d

    for i in range(16):
        if len(left_peak[i]) > 0:
            left_pt_x.append(left_peak[i][np.argmax(left_peak[i])])
            left_pt_y.append((i + 1) * 20 + 10)

    if len(left_pt_y) > 3 and len(left_pt_y) <= 8:
        la, lb = np.polyfit(left_pt_y, left_pt_x, 1)

        for i in range(16):
            y = (i + 1) * 20 + 10
            x = f1(y, la, lb)
            # cv2.circle(left, (int(x),int(y)), 4, (0, 0, 255), -1)
            next_left_pos.append(int(x))

    elif len(left_pt_y) > 8:
        la, lb, lc, ld = np.polyfit(left_pt_y, left_pt_x, 3)

        for i in range(16):
            y = (i + 1) * 20 + 10
            x = f3(y, la, lb, lc, ld)
            # cv2.circle(left, (int(x),int(y)), 4, (0, 0, 255), -1)
            next_left_pos.append(int(x))

    else:
        next_left_pos = [40]*16

    for i in range(16):
        if len(right_peak[i]) > 0:
            right_pt_x.append(right_peak[i][np.argmin(right_peak[i])])
            right_pt_y.append((i + 1) * 20 + 10)

    if len(right_pt_y) > 3 and len(right_pt_y) <= 8:
        ra, rb = np.polyfit(right_pt_y, right_pt_x, 1)
       
        for i in range(16):
            y = (i + 1) * 20 + 10
            x = f1(y, ra, rb,)
            # cv2.circle(right, (int(x),int(y)), 4, (0, 0, 255), -1)
            next_right_pos.append(int(x))

    elif len(right_pt_y) > 8:
        ra, rb, rc, rd = np.polyfit(right_pt_y, right_pt_x, 3)
       
        for i in range(16):
            y = (i + 1) * 20 + 10
            x = f3(y, ra, rb, rc, rd)
            # cv2.circle(right, (int(x),int(y)), 4, (0, 0, 255), -1)
            next_right_pos.append(int(x))

    else:
        next_right_pos = [60]*16

    left_pos = next_left_pos
    right_pos = next_right_pos

    # cv2.imwrite('./avm.jpg', prediction)
    cv2.imshow('frame', frame)
    cv2.imshow('prediction', prediction)
    cv2.waitKey(1)

if __name__ == '__main__':
    left_pos = [40]*16
    right_pos = [60]*16
    bridge = CvBridge()
    model = fcn()
    model.load_weights('./checkpoint/checkpoint-03-0.9650.hdf5', by_name=True)
    graph = tf.get_default_graph()
    rospy.init_node('shopping_assistance_node', anonymous=True)
    rospy.Subscriber('/pub_avm_image', Image, image_callback)
    rospy.spin()
    cv2.destroyAllWindows()

