import cv2 as cv
import matplotlib.pyplot as plt

face_detector = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')  
facemark_detector = cv.face.createFacemarkLBF()
facemark_detector.loadModel('https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml') 


# 传入图片地址,给出人脸的体征点标识
def showFacemark(path):
    mat = cv.imread(path)
    gray_mat = cv.cvtColor(mat, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_mat, 1.05, 5, cv.CASCADE_SCALE_IMAGE, (80, 80))
    _, landmarks = facemark_detector.fit(gray_mat, faces)
    for landmark in landmarks:
        for x, y in landmark[0]:
            # circle坐标需要是整形，浮点会报错
            cv.circle(mat, (int(x), int(y)), 1, (255, 255, 255), 3)
            plt.imshow(mat)

    plt.show()
