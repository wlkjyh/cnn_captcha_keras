from keras.layers import *
from keras.models import *
from keras.optimizers import *
import json
import os
from PIL import Image
from utils import *
from keras import backend as K
from keras.regularizers import l2
import random
# import tensorflow as tf
# load_weights
from keras.models import load_model

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 训练样本
sample_dir = './sample/train/captcha/train'



# 样本的所有标签
char = '0123456789abcdefghijklmnopqrstuvwxyz'

char_len = len(char)
char_set = [str(i) for i in char]

# 验证码长度
captcha_length = 4

# 宽度和高度
width = 160
height = 60

# 批次大小和训练轮次
batch_size = 128
epochs = 1000

train_images_list = os.listdir(sample_dir)
# 打乱
random.shuffle(train_images_list)

label, captcha_array = gen_captcha_text_image(sample_dir,train_images_list[0])

captcha_shape = captcha_array.shape
captcha_shape_len = len(captcha_shape)
if captcha_shape_len == 3:
    image_height, image_width, channel = captcha_shape
elif captcha_shape_len == 2:
    image_height, image_width = captcha_shape
else:
    print('[ERROR] 验证码大小有误')
    os._exit(0)
    
def get_batch(n, size=128):
    # batch_x = np.zeros([size, image_height * image_width])  # 初始化
    batch_x = np.zeros([size, image_height, image_width],dtype=np.uint8)  # 初始化
    batch_y = np.zeros([size, captcha_length * char_len])  # 初始化
    
    for i in range(size):
        label, captcha_array = gen_captcha_text_image(sample_dir,train_images_list[n+i])
        batch_x[i, :] = convert2gray(captcha_array)
        batch_y[i, :] = text2vec(captcha_length,char_len,char_set,label)
        
        print('共计：',len(train_images_list),'当前：',n+i,end='\r')
        
    
    
    return batch_x, batch_y

train_x, train_y = get_batch(0,len(train_images_list))
# print(train_y[0],train_y[1])
# print(train_y[0])
# print(train_x[0].shape)
# exit()
train_x = train_x / 255.0

# print(train_y[0],train_y[1])
# exit()

model = Sequential()

# model.add(Reshape((image_height, image_width, 1), input_shape=(image_height * image_width,)))

# 卷积层1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height, width, 1),kernel_regularizer=l2(0.005)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# 卷积层2
model.add(Conv2D(64, (3, 3), activation='relu',kernel_regularizer=l2(0.005)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# 卷积层3
model.add(Conv2D(128, (3, 3), activation='relu',kernel_regularizer=l2(0.005)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(256, (3, 3), activation='relu',kernel_regularizer=l2(0.005)))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# 全连接层1
model.add(Flatten())
model.add(Dense(1024, activation='relu',kernel_regularizer=l2(0.005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# 全连接层2
model.add(Dense(512, activation='relu',kernel_regularizer=l2(0.005)))
model.add(BatchNormalization())
model.add(Dropout(0.25))


model.add(Dense(144))
model.add(BatchNormalization())


# 结合sigmoid的交叉熵损失函数
def sigmoid_cross_entropy_with_logits(y_true, y_pred):
    x = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return x

# 计算精度
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

model.compile(loss=sigmoid_cross_entropy_with_logits, optimizer=Adam(lr=0.0001), metrics=[char_acc,image_acc])

# model.summary()
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



TensorBoard_CALLBACK = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint(filepath='./weights.h5', monitor='loss', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.05, callbacks=[TensorBoard_CALLBACK, checkpoint, early_stopping])

model.save('./model.h5')