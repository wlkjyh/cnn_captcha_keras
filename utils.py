import numpy as np
import os
from PIL import Image

""" 
    see: https://github.dev/nickliqian/cnn_captcha
"""

def convert2gray(img):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img

def text2vec(max_captcha,char_set_len,char_set,text):
    text_len = len(text)
    if text_len > max_captcha:
        raise ValueError('验证码最长{}个字符'.format(max_captcha))

    vector = np.zeros(max_captcha * char_set_len)

    for i, ch in enumerate(text):
        idx = i * char_set_len + char_set.index(ch)
        vector[idx] = 1
    return vector



def gen_captcha_text_image(img_path, img_name):
    """
        返回一个验证码的array形式和对应的字符串标签
        :return:tuple (str, numpy.array)
    """
        # 标签
    label = img_name.split("_")[0]
        # 文件
    img_file = os.path.join(img_path, img_name)
    captcha_image = Image.open(img_file)
    captcha_array = np.array(captcha_image)  # 向量化
    return label, captcha_array