import os
import hashlib


dirs = os.listdir('./sample/train/captcha/train')

for i in dirs:
    realname = i.split('.')[0]
    # print(realname)
    rename = realname + '_' + hashlib.md5(realname.encode('utf-8')).hexdigest() + '.jpg'
    # print(rename)
    os.rename('./sample/train/captcha/train/' + i, './sample/train/captcha/train/' + rename)