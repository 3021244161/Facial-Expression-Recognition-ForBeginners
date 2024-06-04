import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.applications import VGG19
import cv2

# 用GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU")
    except RuntimeError as e:
        print(e)

# 数据增强
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",  # 设置为单通道灰度图像
    class_mode='categorical'
)

validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",  # 设置为单通道灰度图像
    class_mode='categorical'
)

# 构建 VGG19 模型
inputs = Input(shape=(48, 48, 1))  # 输入为单通道灰度图像
base_model = VGG19(include_top=False, input_tensor=inputs, weights=None)
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(7, activation='softmax')(x)

emotion_model = Model(inputs=inputs, outputs=outputs)

# 编译模型
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# 自定义回调函数
class LossAccuracyLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open('training_log_vgg19.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}: loss = {logs['loss']}, accuracy = {logs['accuracy']}, val_loss = {logs['val_loss']}, val_accuracy = {logs['val_accuracy']}\n")

logger = LossAccuracyLogger()

# 训练模型
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[logger]
)

# 保存模型
model_json = emotion_model.to_json()
with open("emotion_model_vgg19.json", "w") as json_file:
    json_file.write(model_json)

emotion_model.save_weights('emotion_model_vgg19.h5')
