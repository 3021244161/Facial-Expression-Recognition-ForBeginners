import cv2
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, Input, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.applications import ResNet101

# 禁用OpenCL以避免与cv2.ocl.setUseOpenCL(False)冲突
cv2.ocl.setUseOpenCL(False)

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
    rescale=1./255,  # 将像素值归一化到[0,1]
    rotation_range=30,  # 图像随机旋转的角度范围
    width_shift_range=0.2,  # 水平平移的范围
    height_shift_range=0.2,  # 垂直平移的范围
    shear_range=0.2,  # 剪切变换
    zoom_range=0.2,  # 随机缩放
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest'  # 填充新创建像素的方法
)

validation_data_gen = ImageDataGenerator(rescale=1./255)

# 训练数据生成器
train_generator = train_data_gen.flow_from_directory(
    'data/train',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",  # 图像颜色模式为灰度图
    class_mode='categorical'
)

# 验证数据生成器
validation_generator = validation_data_gen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# 构建ResNet-101模型
base_model = ResNet101(include_top=False, weights=None, input_tensor=Input(shape=(48, 48, 1)))

# 获取ResNet-101的输出
x = base_model.output
x = GlobalAveragePooling2D()(x)

# 添加全连接层和输出层
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(7, activation='softmax')(x)

# 定义模型
emotion_model_resnet101 = Model(inputs=base_model.input, outputs=outputs)

# 编译模型
emotion_model_resnet101.compile(loss='categorical_crossentropy',
                                optimizer=Adam(learning_rate=0.0001),
                                metrics=['accuracy'])

# 自定义回调函数
class LossAccuracyLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open('training_log_resnet101.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}: loss = {logs['loss']}, accuracy = {logs['accuracy']}, val_loss = {logs['val_loss']}, val_accuracy = {logs['val_accuracy']}\n")

logger = LossAccuracyLogger()

# 训练模型
emotion_model_info = emotion_model_resnet101.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=100,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[logger]
)

# 保存模型
model_json = emotion_model_resnet101.to_json()
with open("emotion_model_resnet101.json", "w") as json_file:
    json_file.write(model_json)

emotion_model_resnet101.save_weights('emotion_model_resnet101.h5')
