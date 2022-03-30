# 获取感兴趣区域(gray)
def Ori2Standard(path):
    mat = cv.imread(path)
    gray_mat = cv.cvtColor(mat,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(mat,1.05,5,cv.CASCADE_SCALE_IMAGE,(80,80))
    for (x,y,w,h) in faces:
        res = gray_mat[y:y+h,x:x+w] # 这和画矩形不同，注意长和宽的顺序
        return res


# 传入要训练的图片和标签，制作训练集,返回图片集和标签集合
def TrainSet(path, label):
    images = []
    labels = []
    f_list = os.listdir(path)
    for f in f_list:
        # 有时会混入非图片文件，故需要判断
        if f.endswith('png') or f.endswith('jpg'):
            im = Ori2Standard(path + '/' + f)
            # 注意这步很重要，感兴趣区域可能会为空，如果加入了图片集那会让opencv报错，图片数量和标签数量不匹配
            if im is not None:
                images.append(im)
                labels.append(label)
    return images, labels


# 包装了一下predict，可以更方便的输出
def predict(path):
    gray_mat = Ori2Standard(path)
    # confidence表示的实际上是距离，越低越好(之前一直以为自己搞错了，实际上已经训练成功了)
    lable,confidence = recognizer.predict(gray_mat)

    print("预测的lable:" + str(lable))
    print("预测的偏离程度:" + str(confidence))
