# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import math
import time

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import filter_def

def f_p_load_models() :
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'face_part_model'
    LABELMAP_NAME = 'face_part_labelmap'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    model_list = os.listdir(CWD_PATH+"\\"+MODEL_NAME)
    labelmap_list = os.listdir(CWD_PATH+"\\"+LABELMAP_NAME)

    # =========================================================================
    # model 에 따라서 바뀌는 값들 배열 선언
    sess = {}
    detection_graph = {}
    categories = {}
    category_index = {}
    i = 0
    # 모델 불러오는 부분 => 반복 (여러 모델 불러오기)
    for model, labelmap in zip(model_list, labelmap_list):

        # Path to frozen detection graph .pb file, which contains the model that is used for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, model)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH, LABELMAP_NAME, labelmap)

        # Number of classes the object detector can identify
        # max_num_classes 를 지정해주는 값
        NUM_CLASSES = 1

        # Load the label map.
        # Label maps map indices to category names
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        categories[i] = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                    use_display_name=True)
        category_index[i] = label_map_util.create_category_index(categories[i])

        # Load the Tensorflow model into memory.
        # Tensorflow 모델을 메모리에 로드
        # detection_graph 배열로 만들어서 이름 바꿔주면서 image 넣기
        detection_graph[i] = tf.Graph()
        with detection_graph[i].as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            sess[i] = tf.Session(graph=detection_graph[i])
        i = i + 1

    return sess, detection_graph, category_index


def detect_face_part(sess, detection_graph, category_index, folder_path, folder_list) :
    print("filename, img_input_time, process_time, label, percent, label, percent,")
    start_time = time.time()
    eye_list = []
    nose_list = []
    mouth_list = []
    CWD_PATH = os.getcwd()
    model_list = os.listdir(CWD_PATH + "\\" + 'face_part_model')

    for item in folder_list:  # 폴더의 파일이름 얻기
        IMAGE_NAME = folder_path + item
        # Path to image
        PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)

        image = cv2.imread(PATH_TO_IMAGE)
        for i in range(len(sess)):
            model_name = model_list[i].split('.')[0]

            if 'eye' in model_name:
                #     f_image = filter_def.eyebrow_doubleline(image)
                f_image = filter_def.eye(image)
            elif 'nose' in model_name:
                f_image = filter_def.nose(image)
            elif 'mouth' in model_name:
                # f_image = filter_def.mouth_h_b(image)
                f_image = filter_def.mouth(image)
            else:
                f_image = image

            image_expanded = np.expand_dims(f_image, axis=0)
            # Input tensor is the image
            image_tensor = detection_graph[i].get_tensor_by_name('image_tensor:0')
            # Output tensors are the detection boxes, scores, and classes
            # Each box represents a part of the image where a particular object was detected
            detection_boxes = detection_graph[i].get_tensor_by_name('detection_boxes:0')

            # Each score represents level of confidence for each of the objects.
            # The score is shown on the result image, together with the class label.
            detection_scores = detection_graph[i].get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph[i].get_tensor_by_name('detection_classes:0')

            # Number of objects detected
            num_detections = detection_graph[i].get_tensor_by_name('num_detections:0')

            # Perform the actual detection by running the model with the image as input

            # print(str(len(sess))+"개 중에 "+str(i+1)+"번째 모델 통과 중")
            (boxes, scores, classes, num) = sess[i].run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            # Draw the results of the detection (aka 'visulaize the results')
            label_str = ""
            # print("category"+ str(category_index[i]))

            xmin, xmax, ymin, ymax = vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index[i],
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.30)

            if 'eye' in model_name:
                eye_result = [xmin, xmax, ymin, ymax]
                eye_list += [eye_result]
                # print(str('eye_result : ') +str(eye_result))
            elif 'nose' in model_name:
                nose_result = [xmin, xmax, ymin, ymax]
                nose_list += [nose_result]
                # print(str('nose_result : ') + str(nose_result))
            elif 'mouth' in model_name:
                mouth_result = [xmin, xmax, ymin, ymax]
                mouth_list += [mouth_result]
                # print(str('mouth_result : ') + str(mouth_result))

            # print()
            # 현재 박스친 이미지 다시 넘김
            image = image

            load_model_time = time.time() - (start_time)

    return eye_list, nose_list, mouth_list, load_model_time