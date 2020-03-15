import os
import time
from face_part_detect import f_p_load_models, detect_face_part
from fake_detect import load_fake_model, fake_detect

# 모델을 다 불러와서 session 에 올려놓고 시작
model_time = time.time()

# face part 모델 다 불러오기
f_p_sess, f_p_detection_graph, f_p_category_index = f_p_load_models()
# 가짜 특징 모델 다 불러옴
fake_sess, fake_detection_graph, fake_category_index = load_fake_model()

# 폴더 내 이미지 불러오기
folder_path = "nose_gridnoise_v7_ssd\\frontal_original\\"
folder_list = os.listdir(folder_path)
#print(str(folder_list))

# 결과 출력(cv.imwrite)해서 확인하려면 output 경로 수정
output_dir = "nose_gridnoise_v7_ssd\\frontal_original_nose_output\\"

# 데이터 저장
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

eye_list, nose_list, mouth_list, load_model_time = \
    detect_face_part(f_p_sess, f_p_detection_graph, f_p_category_index,
                     folder_path, folder_list)

# print("눈코입 다 찾음")
# 얼굴, 눈, 코, 입 좌표 넘겨서 가짜 특징 찾기
fake_detect(eye_list, nose_list, mouth_list,
            folder_path, folder_list, output_dir,
            fake_sess, fake_detection_graph, fake_category_index)



