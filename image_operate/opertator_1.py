
# 截取原始图像感兴趣区域并转为灰度
def Ori2StandardGray(path):
    mat = cv.cvtColor(cv.imread(path),cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(mat,1.1,5,cv.CASCADE_SCALE_IMAGE,(400,400))
    for (x,y,w,h) in faces:
        mat = mat[y:y+h,x:x+w] # 这和画矩形不同，注意长和宽的顺序
        s_mat = cv.resize(mat,(250,250))
        cv.imwrite("test.jpg",s_mat)
