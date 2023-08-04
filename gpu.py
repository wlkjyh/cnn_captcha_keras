import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print(tf.test.is_gpu_available())
print('>>> 版本：', tf.__version__)
print('>>> 是否支持GPU：', tf.test.is_gpu_available())
