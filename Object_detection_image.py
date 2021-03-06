######## Image Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 1/15/18
# Description: 
# This program uses a TensorFlow-trained neural network to perform object detection.
# It loads the classifier and uses it to perform object detection on an image.
# It draws boxes, scores, and labels around the objects of interest in the image.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.

# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils_image as vis_util
import time
# from fake_filter import filter_def , mtcnn_crop_def , opencv_detect_def

# 코드 시작시간
code_start = time.time()

config= tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session= tf.compat.v1.Session(config=config)

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'models'
LABELMAP_NAME = 'labelmap'

# 테스트할 모델 이름
# model_name = 'thick_temple'

# =========================================================================
# python Object_detection_image.py > nose_bridge_v5_ssd/adapt_filter_a.csv
# 폴더 내 이미지 불러오기"
folder_path = "nose_bridge_v5_ssd\\adapt_filter_b\\"
# folder_path = "G:\\Face_image\\main_original_real\\side_nose_filter\\"
folder_list = os.listdir(folder_path)

# 결과 출력해서 확인하려면 output 경로 수정
output_dir = "nose_bridge_v5_ssd\\adapt_filter_b_output\\"
# output_dir = "F:\\head_cut_output\\"
# 모델, 라벨맵 이름 설정
model = 'nose_bridge_v5_ssd.pb'
label = 'nose_bridge_v5_ssd.pbtxt'

# 출력되는 박스 percentage
percent = 0.1

# =========================================================================
# Grab path to current working directory
CWD_PATH = os.getcwd()

# 모델 불러오는 부분 => 현재 단일 모델 불러옴
# Path to frozen detection graph .pb file, which contains the model that is used for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,model)
#print("model_path : "+PATH_TO_CKPT)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,LABELMAP_NAME,label)
#print("label_path : "+PATH_TO_LABELS)

# # Number of classes the object detector can identify
# max_num_classes 를 지정해주는 값
# 최대 라벨 개수로 지정해주면 될 것 같음
# label class 개수보다 작은 값이 들어가면 N/A 라고 나옴
NUM_CLASSES = 2

# Load the label map.
# Label maps map indices to category names
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

count=0
print("filename, img_input_time, process_time, label, percent, label, percent,")
for item in folder_list:  # 폴더의 파일이름 얻기
    IMAGE_NAME = folder_path+item
    start = time.time()

    # Path to image
    PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    #코드 실행 시작시간
    start_time=time.time() - start

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Load image using OpenCV and
    # expand image dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    image = cv2.imread(PATH_TO_IMAGE)
    h, w, channel = image.shape

    image_expanded = np.expand_dims(image, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_expanded})

    # Draw the results of the detection (aka 'visulaize the results')
    label_str=""
    label_score =0
    image, label_str, label_score= vis_util.visualize_boxes_and_labels_on_image_array(
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=percent)
    label_str = label_str
    # # label 있는 것만 출력
    # if (label_str != ""):
        # print(item + ", " + label_str)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if (label_str != ""):
        count = count + 1
        detect_dir = output_dir + 'detect\\'
        if not os.path.exists(detect_dir):
            os.mkdir(detect_dir)
        cv2.imwrite(detect_dir + item, image)
    else:
        not_detect_dir = output_dir + 'not_detect\\'
        if not os.path.exists(not_detect_dir):
            os.mkdir(not_detect_dir)
        cv2.imwrite(not_detect_dir + item, image)
    print(item+", "+ str(start_time)+", "+str(time.time() - start)+", "+label_str)
    #print("process time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    # # All the results have been drawn on image. Now display the image.
    # # 데이터 확인용
    # # cv2.imshow('Object detector', image)
    # # # Press any key to close the image
    # # cv2.waitKey(0)
    #
    # # 데이터 저장
    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir)
    #
    # cv2.imwrite(output_dir+item, image)

print()
print("code finish time :", time.time() - code_start)  # 현재시각 - 시작시간 = 실행 시간
# Clean up
print(str(len(folder_list))+"개 중 "+str(count)+"개 image detect")
cv2.destroyAllWindows()
