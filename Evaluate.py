# 0 = 愤怒, 1 =厌恶, 2 = 恐惧, 3 = 开心, 4 = 伤心, 5 = 惊讶, 6 = 自然状态
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# 定义情绪字典，对应模型输出的情绪分类
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# 加载模型结构
json_file = open('model_vgg19_100\emotion_model_vgg19.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# 从JSON文件中加载模型架构
emotion_model = model_from_json(loaded_model_json)

# 加载模型权重
emotion_model.load_weights("model_vgg19_100\emotion_model_vgg19.h5")
print("Loaded model from disk")

# 测试数据生成器（仅归一化）
test_data_gen = ImageDataGenerator(rescale=1./255)

# 从目录中生成测试数据
test_generator = test_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical',
        shuffle=False  # 禁止洗牌以便于后续计算混淆矩阵
)

# 使用模型预测测试集
predictions = emotion_model.predict_generator(test_generator)

# 计算混淆矩阵
c_matrix = confusion_matrix(test_generator.classes, predictions.argmax(axis=1))
print("Confusion Matrix:")
print(c_matrix)

# 绘制混淆矩阵图
cm_display = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=list(emotion_dict.values()))
plt.figure(figsize=(10, 8))
cm_display.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.show()

print("-----------------------------------------------------------------")

# 输出分类报告（包括精确度、召回率、F1值等）
print("Classification Report:")
print(classification_report(test_generator.classes, predictions.argmax(axis=1), target_names=list(emotion_dict.values())))




