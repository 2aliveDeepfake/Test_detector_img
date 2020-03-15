import cv2
import numpy as np

# 눈 필터
def eye(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # # Bilateral
    tmp = cv2.bilateralFilter(tmp, 19, 153, 102)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)

    # Unsharp

    G_ksize = 19
    sigmaX = 0
    sigmaY = 0
    alpha = 10
    beta = 5
    gamma = 0

    # 필터는 홀수여야한다.

    if (G_ksize % 2 == 1):
        tmp2 = cv2.GaussianBlur(tmp, (G_ksize, G_ksize), sigmaX / 10, sigmaY / 10)
        work = cv2.addWeighted(tmp, alpha / 10, tmp2, -1 + (beta / 10), gamma)

    # 필터 통과한 이미지 변수에 넣기
    # work = tmp.copy()
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    return work


# 코 필터
def nose(image) :
    tmp = image.copy()

    # Gamma_correction
    gamma = 20
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    #CLAHE
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(tmp)
    clahe = cv2.createCLAHE(clipLimit=0, tileGridSize=(10, 10))
    dst = clahe.apply(l)
    l = dst.copy()
    tmp = cv2.merge((l, a, b))
    tmp = cv2.cvtColor(tmp, cv2.COLOR_LAB2RGB)

    #THRESH_TRUNC
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 180, 255, cv2.THRESH_TRUNC)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    image = tmp.copy()
    return image

# 입 필터
def mouth(image) :
    tmp = image.copy()

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # sharpen (5)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    alpha = 2  # 5로 하면 c코드에서 확인했을 때 보다 sharpen이 많이 적용되는것 같아서 조정함
    kernel_sharpen = np.array([[0, -alpha, 0], [-alpha, 1 + 4 * alpha, -alpha], [0, -alpha, 0]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, kernel_sharpen)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Gaussian_Smoothing
    ksize = 3
    sigmaX = 0
    sigmaY = 0
    tmp = cv2.GaussianBlur(tmp, (ksize, ksize), sigmaX / 10, sigmaY / 10)

    # Bilateral => 3번
    tmp = cv2.bilateralFilter(tmp, 5, 150, 255)
    tmp = cv2.bilateralFilter(tmp, 5, 150, 255)
    tmp = cv2.bilateralFilter(tmp, 5, 150, 255)

    # Roberts
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    # X,Y필터를 만들어준다.
    Kernel_X = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    Kernel_Y = np.array([[0, 0, -1], [0, -1, 0], [0, 0, 0]])
    # 만든필터를 각각 grayscale한 이미지에 적용한다.
    grad_x = cv2.filter2D(tmp, cv2.CV_16S, Kernel_X)
    grad_y = cv2.filter2D(tmp, cv2.CV_16S, Kernel_Y)
    # 픽셀값이 음수가 있을 수 있어서 절대값으로 바꾼다.
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    # 세부변수인 alpha, beta, gamma
    alpha = 100
    beta = 0
    gamma = 0
    # X필터를 통과한 이미지와 Y필터를 통과한 이미지를 일정값을 곱해주고 합친다.
    tmp = cv2.addWeighted(abs_grad_x, alpha / 10, abs_grad_y, beta / 10, gamma / 10)
    # 3채널이미지로 바꿔준다.
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()

    return image

# 얼굴 가로 세로 노이즈
def face_gridnoise(image) :
    tmp = image.copy()
    #Sharpen
    ksize_sharp = 12
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_sharpen = np.array([[0, -ksize_sharp, 0], [-ksize_sharp, 1+4*ksize_sharp, -ksize_sharp], [0, -ksize_sharp, 0]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_sharpen)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    #Gaussian_Smoothing
    tmp = cv2.GaussianBlur(tmp, (7,7), 0,1)
    image = tmp.copy()
    return image

# 얼굴 점 노이즈
def face_dotnoise(image) :
    tmp = image.copy()

    # 학습 진행 필터
    # Emboss
    tmp = tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel)
    # Bilateral
    tmp = cv2.bilateralFilter(tmp, 4, 71, 43)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    image = tmp.copy()
    return image

# 눈썹에 세로 선
def eyebrow_vertical_line(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # THRESH_BINARY
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 95, 255, cv2.THRESH_BINARY)

    ###### could not broadcast input array from shape 에러시
    # 차원 관련 문제니 gray2rgb 적용
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()
    return image


# 미간에 콧구멍
def nose_in_b(image) :
    tmp = image.copy()
    # THRESH_BINARY
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    ret, tmp = cv2.threshold(tmp, 55, 255, cv2.THRESH_BINARY)

    ###### could not broadcast input array from shape 에러시
    # 차원 관련 문제니 gray2rgb 적용
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()
    return image


# 코 노이즈
def nose_noise(image) :
    tmp = image.copy()

    gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    # # #emboss필터 적용
    tmp = gray.copy()
    # work = tmp.copy()
    Kernel = np.array([[-11, -1, 0], [-1, 1, 1], [0, 1, 11]])
    work = cv2.filter2D(tmp, cv2.CV_8U, Kernel)
    work = cv2.cvtColor(work, cv2.COLOR_GRAY2RGB)

    # 필터 통과한 이미지 변수에 넣기
    image = work.copy()

    return image

# 입 아래 가로 검은 선
def mouth_h_b(image) :
    tmp = image.copy()

    # Emboss
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    Kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    tmp = cv2.filter2D(tmp, cv2.CV_8U, Kernel_emboss)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Histogram_Equalization
    tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2GRAY)
    cv2.equalizeHist(tmp, tmp)
    tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)

    # Gamma_correction
    gamma = 60
    lookUpTable = np.empty((1, 256), np.uint8)
    for j in range(256):
        lookUpTable[0, j] = np.clip(pow(j / 255.0, 1.0 / (gamma / 10)) * 255.0, 0, 255)

    tmp = cv2.LUT(tmp, lookUpTable)

    # Bilateral
    tmp = cv2.bilateralFilter(tmp, 13, 51, 51)

    # Roberts
    Kernel_X = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])
    Kernel_Y = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])

    grad_x = cv2.filter2D(tmp, cv2.CV_16S, Kernel_X)
    grad_y = cv2.filter2D(tmp, cv2.CV_16S, Kernel_Y)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    tmp = cv2.addWeighted(abs_grad_x, 10, abs_grad_y, 10, 0)

    tmp.astype(np.uint8)

    # cv2.imshow("bbb",tmp)
    # cv2.waitKey()

    # 필터 통과한 이미지 변수에 넣기
    image = tmp.copy()
    return image
