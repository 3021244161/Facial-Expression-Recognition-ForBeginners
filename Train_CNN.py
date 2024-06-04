import cv2
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback #回调函数

# 用GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU")
    except RuntimeError as e:
        print(e)

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# 数据增强
train_data_gen = ImageDataGenerator(
    rescale=1./255,  # 将像素值归一化到[0,1]
    rotation_range=30,  # 图像随机旋转的角度范围
    width_shift_range=0.2,  #水平平移的范围
    height_shift_range=0.2,  # 垂直平移的范围
    shear_range=0.2, #剪切变换
    zoom_range=0.2,  #  随机缩放
    horizontal_flip=True,  #水平翻转
    fill_mode='nearest'  # 填充新创建像素的方法
)


train_generator = train_data_gen.flow_from_directory(  #创建训练数据生成器
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale", #图像颜色模式为灰度图
        class_mode='categorical')

# 验证数据生成器
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

emotion_model = Sequential()

# 第一卷积层组
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# 第二卷积层组
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

# 第三卷积层组
emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
emotion_model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())#全连接层
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5)) # 防止过拟合
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# 自定义回调函数
class LossAccuracyLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open('training_log.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}: loss = {logs['loss']}, accuracy = {logs['accuracy']}, val_loss = {logs['val_loss']}, val_accuracy = {logs['val_accuracy']}\n")

logger = LossAccuracyLogger()

emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=29429 // 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=7360 // 64,
        callbacks=[logger])

model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

emotion_model.save_weights('emotion_model.h5')
