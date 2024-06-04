# EmotionDetecter


## 实验环境

请务必保证tensorflow-gpu、CUDA与cuDNN的版本兼容

- python == 3.9.7
- numpy = 1.24.4
- opencv == 4.6.0
- h5py == 3.6.0
- tensorflow-gpu == 2.9.0
- CUDA ==11.5.2
- cuDNN == 8.3.2
- scipy == 1.13.1


## 数据集

本实验使用的数据集为CK+与FER-2013混合的数据集，共33218张48x48像素的灰度人脸图像，其中训练数据29445张，测试数据3773张

标签(0=愤怒、1=厌恶、2=恐惧、3=快乐、4=悲伤、5=惊讶、6=中性)

- CK+ (https://www.kaggle.com/datasets/davilsena/ckdataset)
- FER-2013 (https://www.kaggle.com/datasets/msambare/fer2013)

## 说明

- 数据准备就绪运行Train_X.py文件训练模型，生成log.txt，运行show_AL.py可将训练log可视化
- 运行Evaluate.py测试模型生成混淆矩阵，输出正确率
- 模型保存后运行TestEmotionDetector.py导入训练好的模型调用摄像头进行实时表情识别

tju ll  2024.5.26