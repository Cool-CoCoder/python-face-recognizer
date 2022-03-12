# 截取原始图像感兴趣区域并覆盖源文件
def Ori2Standard(path):
    mat = cv.imread(path)
    gray_mat = cv.cvtColor(mat,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(mat,1.1,5,cv.CASCADE_SCALE_IMAGE,(400,400))
    for (x,y,w,h) in faces:
        mat = mat[y:y+h,x:x+w] # 这和画矩形不同，注意长和宽的顺序
        s_mat = cv.resize(mat,(250,250))
        cv.imwrite(path,s_mat)


# 清洗文件夹的图片，清洗整个文件夹图片
def File2Standard(path):
    f_list = os.listdir(path)
    for f in f_list:
        Ori2Standard(path+'/'+f)

# 传入要训练的图片和标签，制作训练集,返回图片集和标签集合
def TrainSet(path,label):
    images = []
    labels = []
    f_list = os.listdir(path)
    for f in f_list:
        mat = cv.imread(path + '/' + f)
        gray_mat = cv.cvtColor(mat,cv.COLOR_BGR2GRAY)
        images.append(gray_mat)
        labels.append(label)
    return images,labels
