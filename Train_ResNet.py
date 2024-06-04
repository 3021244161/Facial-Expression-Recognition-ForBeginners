import cv2
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, BatchNormalization, Activation, Add
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import Callback
from keras.regularizers import l2

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

# 定义ResNet残差块
def residual_block(x, filters, kernel_size=(3, 3), strides=(1, 1), activation='relu'):
    y = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.01))(x)
    y = BatchNormalization()(y)
    y = Activation(activation)(y)
    
    y = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.01))(y)
    y = BatchNormalization()(y)
    y = Activation(activation)(y)
    
    y = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_regularizer=l2(0.01))(y)
    y = BatchNormalization()(y)

    # Skip connection
    if x.shape[-1] != filters:
        x = Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(x)

    y = Add()([x, y])
    y = Activation(activation)(y)
    return y

# 构建ResNet模型
inputs = Input(shape=(48, 48, 1))
x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)

# 堆叠残差块
x = residual_block(x, filters=64)
x = residual_block(x, filters=64)
x = residual_block(x, filters=64)

x = residual_block(x, filters=128)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = residual_block(x, filters=256)
x = MaxPooling2D(pool_size=(2, 2))(x)

# 全连接层
x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
outputs = Dense(7, activation='softmax')(x)

# 定义模型
emotion_model_resnet = Model(inputs=inputs, outputs=outputs)

# 编译模型
emotion_model_resnet.compile(loss='categorical_crossentropy',
                             optimizer=Adam(learning_rate=0.0001),
                             metrics=['accuracy'])

# 自定义回调函数
class LossAccuracyLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open('training_log_resnet.txt', 'a') as f:
            f.write(f"Epoch {epoch + 1}: loss = {logs['loss']}, accuracy = {logs['accuracy']}, val_loss = {logs['val_loss']}, val_accuracy = {logs['val_accuracy']}\n")

logger = LossAccuracyLogger()

# 训练模型
emotion_model_info = emotion_model_resnet.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[logger]
)

# 保存模型
model_json = emotion_model_resnet.to_json()
with open("emotion_model_resnet.json", "w") as json_file:
    json_file.write(model_json)

emotion_model_resnet.save_weights("emotion_model_resnet.h5")
