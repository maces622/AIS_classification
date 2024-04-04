import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
import keras
from keras import layers

import pickle as pkl
import matplotlib.pyplot as plt
import math
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    # 重参数技巧，因为采样过程无法进行梯度下降
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

"""
global settings
"""
# 隐藏空间维度
latent_dim = 3
# shape : 输入形状 长 宽 通道数目
# AIS数据重新采样后长度为 180（暂定），字段数目为 10 ，独热码形式通道为1(gray graph 1,RGB 3)
height = 160
width  = 80
chan = 1
beta=0.4
input_shape = (height, width,1)  # 对于灰度图像，通道数是1
# number of vessel types
num_classes = 4
"""
^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

"""
Build the classifier=========================
"""
# 输入层
input_layer=keras.Input(shape=input_shape)

# 卷积层堆叠
x = layers.Conv2D(1, (10, 10), activation='relu', strides=2, padding='same')(input_layer)
x = layers.Conv2D(5, (10, 10), activation='relu', strides=2, padding='same')(x)
x = layers.Conv2D(5, (10, 10), activation='relu', strides=2, padding='same')(x)
x = layers.Conv2D(5, (5, 5), activation='relu', strides=2, padding='same')(x)
x = layers.Conv2D(5, (3, 3), activation='relu', strides=1, padding='same')(x)

# 扁平化后接全连接层
x = layers.Flatten()(x)
x = layers.Dense(250, activation='relu')(x)

# 输出层
output_layer = layers.Dense(num_classes, activation='softmax')(x)  # num_classes是输出的分类数，根据实际情况替换

# 创建模型
classifier = keras.Model(inputs=input_layer, outputs=output_layer)

# 模型概要
classifier.summary()
"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""



"""
Build the encoder============================
"""


# 输入层：假设我们已经有了处理好的轨迹数据和标签
label_shape = (num_classes,)  # 标签的形状，使用one-hot编码

# 轨迹输入
x_input = keras.Input(shape=input_shape, name='input_trajectory')

# 标签输入
label_input = keras.Input(shape=label_shape, name='input_label')

# 卷积块
x = layers.Conv2D(1, (10, 10), activation='relu', strides=2, padding='same')(x_input)
x = layers.Conv2D(5, (10, 10), activation='relu', strides=2, padding='same')(x)
x = layers.Conv2D(5, (10, 10), activation='relu', strides=2, padding='same')(x)
x = layers.Conv2D(5, (5, 5), activation='relu', strides=2, padding='same')(x)
x = layers.Conv2D(5, (3, 3), activation='relu', strides=1, padding='same')(x)

x = layers.Flatten()(x)

# 全连接层获取轨迹特征
x_features = layers.Dense(250, activation='relu')(x)

# 全连接层获取标签特征
label_features = layers.Dense(50, activation='relu')(label_input)

# 特征拼接
concatenated_features = layers.concatenate([x_features, label_features])

# 两个全连接层获取变量 μ 和 σ
z_mean = layers.Dense(latent_dim, name='z_mean')(concatenated_features)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(concatenated_features)

# 用 μ 和 σ 通过重参数化技术得到潜在变量 z
z = Sampling()([z_mean, z_log_var])

# 创建编码器模型
encoder = keras.Model(inputs=[x_input, label_input], outputs=[z_mean, z_log_var, z], name='encoder')

# 模型概要
encoder.summary()
"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""



"""
Build the decoder============================
"""
# 假设潜在变量z的维度和标签y的one-hot编码维度已知

num_classes = 4 # 分类的数量，即one-hot编码的维度

# 潜在变量和标签的输入层
z_input = keras.Input(shape=(latent_dim,), name='input_latent')
label_input = keras.Input(shape=(num_classes,), name='input_label')

# 特征拼接
concatenated_features = layers.concatenate([z_input, label_input])

# 通过两个全连接层生成特征向量
x = layers.Dense(250, activation='relu')(concatenated_features)
x = layers.Dense(250, activation='relu')(x)

# 将特征向量恢复到合适的维度以便进行上采样
x = layers.Reshape((10, 5, 5))(x) # 这里的目标维度取决于您想要如何开始上采样过程

# 五个反卷积层（上采样层），使用给定的输出通道数和内核大小
x = layers.Conv2DTranspose(5, kernel_size=(3, 3), activation='relu', strides=1,padding='same')(x)
x = layers.Conv2DTranspose(5, kernel_size=(5, 5), activation='relu',  strides=2,padding='same')(x)
x = layers.Conv2DTranspose(5, kernel_size=(10, 10), activation='relu',  strides=2,padding='same')(x)
x = layers.Conv2DTranspose(5, kernel_size=(10, 10), activation='relu',  strides=2,padding='same')(x)
x_output = layers.Conv2DTranspose(1, kernel_size=(10, 10), activation='sigmoid',  strides=2,padding='same')(x)

# 最后一个反卷积层生成原始轨迹，使用sigmoid激活函数
# 创建解码器模型
decoder = keras.Model(inputs=[z_input, label_input], outputs=x_output, name='decoder')

# 模型概要
decoder.summary()
"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""



"""
compucate the value of loss functions========
"""
def compute_loss_labeled(encoder, decoder, x, y):
    # Encode the input to get the mean, log-variance, and sampled z
    z_mean, z_log_var, z = encoder([x, y])
    # Decode the sampled z to reconstruct the image
    reconstruction = decoder([z, y])
    
    # Compute the binary cross-entropy for reconstruction loss
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(keras.losses.binary_crossentropy(x,reconstruction),axis=(1,2))
    )
    
    # Compute the KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
    )
    
    # Sum the two loss terms
    loss = reconstruction_loss + kl_loss
    return loss,kl_loss
batch_size=40
def compute_loss_unlabeled(encoder, decoder, classifier, x):
    # Sum over all possible labels
    u_loss = 0
    all_possible_labels=[0,1,2,3]

    predict_res=classifier(x)
    # tf.print(predict_res)
    ## 船只种类
    for y in all_possible_labels:  # all_possible_labels should be one-hot encoded labels
        arrays = np.full((batch_size,), y)
        label_1 = np.eye(4)[arrays]
        label_1 = np.expand_dims(label_1, -1)
        y_tensor = tf.convert_to_tensor(label_1)


        loss,kl_loss = compute_loss_labeled(encoder, decoder, x, y_tensor)
        q_y_given_x=predict_res[:,y]
        u_loss += q_y_given_x * (-loss)  # q_y_given_x should be the predicted probability of label y
          
    # Add the entropy of the predicted label distribution
    entropy_loss = -tf.reduce_sum(predict_res * tf.math.log(predict_res + tf.keras.backend.epsilon()), axis=-1)
    # Sum the negative lower bound and entropy loss
    total_u_loss = u_loss + entropy_loss
    return total_u_loss
def compute_classifier_loss(classifier, x_labeled, y_true):
    # Classifier predictions
    y_pred = classifier(x_labeled)
    
    # Compute the cross-entropy loss
    y_true_squeezed = tf.squeeze(y_true, axis=-1)
    clf_loss = tf.reduce_mean(
        keras.losses.categorical_crossentropy(y_true_squeezed, y_pred)

    )
    
    return clf_loss




"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
class VAE(keras.Model):

    def __init__(self, encoder,decoder,clf,**kwargs):
        # 调用父类的__init__()
        super().__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder
        self.clf=clf
        self.total_loss_tracker = keras.metrics.Mean(name="Loss")
        self.l1_loss_tracker = keras.metrics.Mean(name="Loss_1")
        self.l2_loss_tracker = keras.metrics.Mean(name="Loss_2")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.clf_loss_tracker= keras.metrics.Mean(name="Loss_clf")
        self.alpha_tracker=keras.metrics.Mean(name="alpha")
# train step每次会自动调取数据集
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.l1_loss_tracker,
            self.l2_loss_tracker,
            self.kl_loss_tracker,
            self.clf_loss_tracker,
            self.alpha_tracker,
        ]
    def train_step(self,data):
        [x_supervised, y_supervised],x_unsupervised = data
        # print(x_supervised.shape,y_supervised.shape,x_unsupervised.shape)
        # x_unsupervised=x_supervised
        alpha=beta*float(0.5)
        # print(x_supervised.shape,y_supervised.shape)
        with tf.GradientTape() as Tape:
            # 均值、方差、z

            Loss_1,kl_loss=compute_loss_labeled(self.encoder,self.decoder,x_supervised,y_supervised)
            Loss_2=compute_loss_unlabeled(self.encoder,self.decoder,self.clf,x_unsupervised)
            Loss_clf=compute_classifier_loss(self.clf,x_labeled=x_supervised,y_true=y_supervised)
            Loss=Loss_1+Loss_2+alpha*Loss_clf

        grads = Tape.gradient(Loss, self.trainable_weights)

        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(Loss)
        self.l1_loss_tracker.update_state(Loss_1)
        self.l2_loss_tracker.update_state(Loss_2)
        self.clf_loss_tracker.update_state(Loss_clf)
        self.kl_loss_tracker.update_state(kl_loss)
        self.alpha_tracker.update_state(alpha)
        return {
            "loss": self.total_loss_tracker.result(),
            "Loss_1": self.l1_loss_tracker.result(),
            "Loss_2": self.l2_loss_tracker.result(),
            "kl_loss":self.kl_loss_tracker.result(),
            "Loss_clf": self.clf_loss_tracker.result(),
            "alpha": self.alpha_tracker.result(),
        }
"""AIS stream to 7-hot code"""
def convert_to_one_hot(data):
    # 获取数据的形状
    n, _, channels = data.shape
    # 初始化一个用于存储结果的数组
    total_bins=80
    result = np.zeros((n, 160, total_bins))

    # 遍历每一个数据样本
    for i in range(n):
        # 遍历每一个通道
        channel_4_data=data[i,:,4]
        channel_5_data=data[i,:,5]
        channel_6_data=data[i,:,6]
        for j in range(2):
            channel_data = data[i, :, j]
            one_hot=np.eye(8)[np.round(channel_data*7).astype(int)]
            result[i,:,j*8:(j+1)*8]=one_hot
        for j in range(2):
            channel_data = data[i, :, j+2]
            one_hot=np.eye(20)[np.round(channel_data*19).astype(int)]
            result[i,:,(j+2)*20:(j+2+1)*20]=one_hot

        one_hot_4=np.eye(8)[np.round(channel_4_data*7).astype(int)]
        one_hot_5=np.eye(8)[np.round(channel_5_data*7).astype(int)]
        one_hot_6=np.eye(8)[np.round(channel_6_data*7).astype(int)]
        result[i, :, 4 * 8:(4 + 1) * 8] = one_hot_4
        result[i, :, 5 * 8:(5 + 1) * 8] = one_hot_5
        result[i, :, 6 * 8:(6 + 1) * 8] = one_hot_6


    return result

with open('procd_data.pkl','rb') as f:
    data = pkl.load(f)


"""
将数据集分为训练集和测试集，
并将AIS字段和其船只类型标签拆开
"""

train_end=int(len(data)*0.8)
# print(train_end)

"""
# [0 mmsi,1 TS,2 lat,3 lon,4 sog,5 cog,
# 6 length,7 width,8 draft,9vesselType]
"""




"""
input shape 
train set
data  x:(286, 180, 9)
label y:(286,)

test set


# (x1_train, t1arget_train), (x1_test, t1arget_test) = keras.datasets.mnist.load_data()

# print(x_train.shape)
# print(target_train.shape)
"""

# 接下来需要进行归一化处理
# 首先需要计算出需要归一化的字段的数据范围。
max_lat=-19999.0
max_lon=-19999.0
min_lat=19999.0
min_lon=19999.0
max_sog=0.0
min_sog=2000.0
max_cog=0.0
min_cog=360.0

max_len=0.0
min_len=10000.0
max_wid=0.0
min_wid=10000.0
max_drt=0.0
min_drt=10000.0
"""
# [0 mmsi,1 TS,2 lat,3 lon,4 sog,5 cog,
# 6 length,7 width,8 draft,9vesselType]
"""
for track in data:
    for row in track:
        max_lat=max(max_lat,row[2])
        min_lat=min(min_lat,row[2])
        max_lon=max(max_lon,row[3])
        min_lon=min(min_lon,row[3])
        max_sog=max(max_sog,row[4])
        min_sog=min(min_sog,row[4])
        max_cog=max(max_cog,row[5])
        min_cog=min(min_cog,row[5])

        max_len=max(max_len,row[6])
        min_len=min(min_len,row[6])
        max_wid=max(max_wid,row[7])
        min_wid=min(min_wid,row[7])
        max_drt=max(max_drt,row[8])
        max_drt=min(max_drt,row[8])

data[:,:,2]=(data[:,:,2]-min_lat)/(max_lat-min_lat)
data[:,:,3]=(data[:,:,3]-min_lon)/(max_lon-min_lon)
data[:,:,4]=(data[:,:,4]-min_sog)/(max_sog-min_sog)
data[:,:,5]=(data[:,:,5]-min_cog)/(max_cog-min_cog)

data[:,:,6]=(data[:,:,6]-min_len)/(max_len-min_len)
data[:,:,7]=(data[:,:,7]-min_wid)/(max_wid-min_wid)
data[:,:,8]=(data[:,:,8]-min_drt)/(max_drt-min_drt)

_train=data[:200].copy()
_test=data[200:400].copy()


x_train=_train[:,:,[2,3,4,5,6,7,8]].astype(float)
target_train=_train[:,0,9].astype(int)

x_test=_test[:,:,[2,3,4,5,6,7,8]].astype(float)
target_test=_test[:,0,9].astype(int)
# 7 8 6 3
# print(target_train.shape)
# change label to one-hot format
conditions1 = [
    target_train == 3,
    target_train == 6,
    target_train == 7,
    target_train == 8,
]
conditions2 = [
    target_test == 3,
    target_test == 6,
    target_test == 7,
    target_test == 8,
]
# 定义对应条件下的替换值
choices = [0, 1, 2, 3]

# 应用 numpy.select

arr_new = np.select(conditions1, choices, default=target_train)
target_train_1=np.eye(4)[arr_new]
arr_new1 = np.select(conditions2, choices, default=target_test)
target_test_1=np.eye(4)[arr_new1]

x_train=convert_to_one_hot(x_train)
x_test=convert_to_one_hot(x_test)
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)
target_train_1 = np.expand_dims(target_train_1,-1)

print(x_train.shape)

vae = VAE(encoder, decoder,classifier)
vae.compile(optimizer=keras.optimizers.Adam())

vae.fit([x_train,target_train_1],x_test, epochs=10, batch_size=40)
def plot_label_clusters(encoder, decoder, x_train, target_train):
    # Display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict([x_test,target_test_1])
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')  # 创建一个三维的绘图工具
    ax.scatter(z_mean[:200, 0], z_mean[:200, 1], z_mean[:200, 2], c=arr_new1[:200])
    # plt.colorbar()
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_zlabel("z[2]")
    plt.show()



plot_label_clusters(encoder, decoder, x_train, target_train_1)

