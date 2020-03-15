# Fake_detector_test
  
  
    
  model 폴더에 pb 파일추가  
  labelmap 폴더에 pbtxt 파일추가
   
  ### 단일 모델을 테스트 하는 경우  
    
  ※테스트 데이터 필터 거쳐서 넣어야함  
    
  python Object_detection_image.py 실행  
    
  csv 파일 만들려면  
  python Object_detection_image.py > 만들파일이름.csv 실행  
    
  ### 눈코입 모델을 거쳐서 영역 내에서 가짜 특징 모델을 테스트 하는 경우  
    
  ※테스트 데이터 원본으로 넣어야함  
    
  python Main.py 실행
    
  csv 파일 만들려면  
  python Main.py > 만들파일이름.csv 실행
