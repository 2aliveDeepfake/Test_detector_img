# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import math
import time

# Import utilites
from utils import label_map_util
from utils import visualization_utils_fake as vis_util
import filter_def

def load_fake_model() :
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'models'
    LABELMAP_NAME = 'labelmap'

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
        # print(str(len(model_list))+"개 중에 "+str(i+fix_keep)+"번째 모델 불러오는 중")
        # print(model+", "+labelmap)

        # Path to frozen detection graph .pb file, which contains the model that is used for object detection.
        PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, model)
        # print("model_path : "+PATH_TO_CKPT)

        # Path to label map file
        PATH_TO_LABELS = os.path.join(CWD_PATH, LABELMAP_NAME, labelmap)
        # print("label_path : "+PATH_TO_LABELS)

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
    # print("모델 불러오는데 걸리는 시간 : "+str(round(time.time() - code_start,5)))
    # ==========================================================================

    return sess, detection_graph, category_index


def fake_detect(eye_list, nose_list, mouth_list, folder_path, folder_list, output_dir,
            fake_sess, fake_detection_graph, fake_category_index) :
    code_start = time.time()

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'models'
    # Grab path to current working directory
    CWD_PATH = os.getcwd()
    model_list = os.listdir(CWD_PATH + "\\" + MODEL_NAME)

    e_list = eye_list
    n_list = nose_list
    m_list = mouth_list
    # part_list = []
    for item in folder_list:  # 폴더의 파일이름 얻기
        start = time.time()
        IMAGE_NAME = folder_path + item
        # Path to image
        PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)
        # 이미지 넣은 시간
        # input_time = time.time()
        count = 0
        # print(PATH_TO_IMAGE)
        image = cv2.imread(PATH_TO_IMAGE)
        start_time = time.time() - start
        for i in range(len(fake_sess)):
            model_name = model_list[i].split('.')[0]
            # print(model_name)
            if 'face' in model_name:
                if 'grid' in model_name:
                    f_image = filter_def.face_gridnoise(image)
                elif 'dot' in model_name:
                    f_image = filter_def.face_dotnoise(image)
            # elif 'eye' in model_name:
            #     if 'line_1' in model_name:
            #         f_image = filter_def.eyebrow_vertical_line(image)
            elif 'nose' in model_name:
                if 'noise' in model_name:
                    f_image = filter_def.nose_noise(image)
                # elif 'in_b' in model_name:
                #     f_image = filter_def.nose_in_b(image)
            # elif 'mouth' in model_name:
            #     if 'ul' in model_name:
            #         f_image = filter_def.mouth_h_b(image)
            else:
                f_image = image

            image_expanded = np.expand_dims(f_image, axis=0)

            # Input tensor is the image
            image_tensor = fake_detection_graph[i].get_tensor_by_name('image_tensor:0')

            # Output tensors are the detection boxes, scores, and classes
            # Each box represents a part of the image where a particular object was detected
            detection_boxes = fake_detection_graph[i].get_tensor_by_name('detection_boxes:0')

            # Each score represents level of confidence for each of the objects.
            # The score is shown on the result image, together with the class label.
            detection_scores = fake_detection_graph[i].get_tensor_by_name('detection_scores:0')
            detection_classes = fake_detection_graph[i].get_tensor_by_name('detection_classes:0')

            # Number of objects detected
            num_detections = fake_detection_graph[i].get_tensor_by_name('num_detections:0')

            # Perform the actual detection by running the model with the image as input

            # print(str(len(sess))+"개 중에 "+str(i+1)+"번째 모델 통과 중")
            (boxes, scores, classes, num) = fake_sess[i].run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_expanded})

            # Draw the results of the detection (aka 'visulaize the results')
            label_str = ""

            if 'eye' in model_name:
                part_list = e_list[count]
            elif 'nose' in model_name:
                part_list = n_list[count]
            elif 'mouth' in model_name:
                part_list = m_list[count]
            else : part_list = (0,1,0,1)

            label_str = vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                part_list,
                model_name,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                fake_category_index[i],
                use_normalized_coordinates=True,
                line_thickness=2,
                min_score_thresh=0.30)

            # 현재 박스친 이미지 다시 넘김
            label_str = label_str
            # # label 있는 것만 출력
            # if(label_str != "") :
            #     print(item + ", " + label_str)
            # print(item + ", " + str(round(time.time(), 5)) + ", " + str(
            #     round(time.time() - (input_time), 5)) + ", " + label_str)
            # print(item + ", " + str(round(start_time,5)) + ", " + str(round(time.time() - start,5)) + ", " + label_str)
            count += 1
            if count >= len(eye_list) \
                    or count >= len(nose_list)\
                    or count >= len(mouth_list): break

        if (label_str != None):
            # count = count + 1
            detect_dir = output_dir + 'detect\\'
            if not os.path.exists(detect_dir):
                os.mkdir(detect_dir)
            cv2.imwrite(detect_dir + item, image)
            print(item + ", " + str(start_time) + ", " + str(time.time() - start) + ", " + str(label_str))
        else:
            not_detect_dir = output_dir + 'not_detect\\'
            if not os.path.exists(not_detect_dir):
                os.mkdir(not_detect_dir)
            cv2.imwrite(not_detect_dir + item, image)
            print(item + ", " + str(start_time) + ", " + str(time.time() - start) + ", " )


        # cv2.imwrite(output_dir + item, image)
