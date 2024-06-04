import cv2
import numpy as np
from keras.models import model_from_json

# 定义情绪字典，对应模型输出的情绪分类
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#加载模型结构

json_file = open('model_vgg19_100\emotion_model_vgg19.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
#   从JSON文件中加载模型架构
emotion_model = model_from_json(loaded_model_json)


emotion_model.load_weights("model_vgg19_100\emotion_model_vgg19.h5")
print("Loaded model from disk")

#使用摄像头进行实时情绪检测
cap = cv2.VideoCapture(0)
# 也可以使用视频文件
# cap = cv2.VideoCapture(r"D:\\计算机视觉大作业\\bqsb\\test.mp4")

while True:
    # 读取帧
    ret, frame = cap.read()
    try:
        # 调整帧大小为 640x480
        frame = cv2.resize(frame, (640, 480))
    except:
        break

    if not ret:
        break

    # 加载Haar级联分类器用于人脸检测
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    # 将帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # 遍历检测到的人脸
    for (x, y, w, h) in num_faces:
        # 在人脸区域绘制矩形框，颜色为白色，宽度为2
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 255, 255), 2)
        # 提取人脸区域并调整大小为 48x48 像素
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # 使用情绪检测模型进行预测
        emotion_prediction = emotion_model.predict(cropped_img)
        # 获取预测结果中概率最大的索引
        maxindex = int(np.argmax(emotion_prediction))
        # 在帧上绘制预测的情绪类别
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # 显示处理后的帧
    cv2.imshow('Emotion Detection', frame)
    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
