from keras.models import load_model
from PIL import Image
import numpy as np
import os
from utils import *
import tensorflow as tf

def sigmoid_cross_entropy_with_logits(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

 
char = '0123456789abcdefghijklmnopqrstuvwxyz'
captcha_length = 4
char_len = len(char)




def sigmoid_cross_entropy_with_logits(y_true, y_pred):
    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def char_acc(y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, captcha_length, char_len])  # 预测结果
    max_idx_p = tf.argmax(predict, 2)  # 预测结果
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, captcha_length, char_len]), 2)  # 标签
        # 计算准确率
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # accuracy_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))
    
    return accuracy_char_count

def image_acc(y_true, y_pred):
    predict = tf.reshape(y_pred, [-1, captcha_length, char_len])  # 预测结果
    max_idx_p = tf.argmax(predict, 2)  # 预测结果
    max_idx_l = tf.argmax(tf.reshape(y_true, [-1, captcha_length, char_len]), 2)  # 标签
        # 计算准确率
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    # accuracy_char_count = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    accuracy_image_count = tf.reduce_mean(tf.reduce_min(tf.cast(correct_pred, tf.float32), axis=1))
    
    return accuracy_image_count




char_len = len(char)
char_set = [str(i) for i in char]

import random
model = load_model('./weights.h5', custom_objects={'sigmoid_cross_entropy_with_logits': sigmoid_cross_entropy_with_logits, 'char_acc': char_acc, 'image_acc': image_acc})

# model = load_model('./model.h5')
while True:
    dirs = os.listdir('./sample/train/captcha/test')
    files = random.choice(dirs)
    labels = files.split('/')[-1].split('.')[0].split('_')[0]
    image = Image.open('./sample/train/captcha/test/' + files)
    image_array = np.array(image)
    image_array = convert2gray(image_array)
    image_array = image_array / 255.0
    image_array = image_array.reshape((1, 60, 160, 1))
    predict = model.predict(image_array)
   
    
    real_value = list(labels)
    
    # 先将预测值分成4个部分
    predict_value = np.split(predict, 4, axis=1)
    # 对每个部分进行sigmoid处理
    for i in range(len(predict_value)):
        predict_value[i] = tf.nn.sigmoid(predict_value[i])
    # print(predict_value)
    
    # 输出每个部分的最大值
    for i in range(len(predict_value)):
        predict_value[i] = tf.argmax(predict_value[i], 1)
        
    predict_value = np.array(predict_value).reshape(-1)
    print('>>> 真实值：', real_value, '>>> 预测值：', predict_value)