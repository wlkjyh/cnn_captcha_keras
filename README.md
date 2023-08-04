# cnn_captcha_keras
在keras上实现的cnn神经卷积网络的验证码识别



### 如何使用？
创建3.10python环境
```shell
conda create -n cnn_captcha_keras python=3.10
```
进入环境
```shell
conda activate cnn_captcha_keras
```
安装必要的包
```shell
pip3 install -r requirement.txt
```

配置train.py中相关参数
```py
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


```


将样本放入sample_dir所在的目录


运行train.py
```
python train.py
```



### 如何预测？
请参考test.py的写法，你可以直接运行test.py，需要在这个文件中配置测试集的路径
```
python test.py
```



### 准确率如何？
在85轮迭代后准确率可以到达90，110轮达到91%，300轮以后基本上可以达到98%以上
