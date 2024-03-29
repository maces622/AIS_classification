import tensorflow as tf
import numpy as np
import keras
from keras import layers
from keras.callbacks import Callback, EarlyStopping
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.callbacks import LambdaCallback, TensorBoard
from keras import models

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
latent_dim = 10
# shape : 输入形状 长 宽 通道数目
# AIS数据重新采样后长度为 160（暂定），字段数目为 7 ，独热码形式通道为1
height = 160
width  = 8

beta=0.4

"""
global settings
"""
input_shape = (height, width, 1)  # 对于灰度图像，通道数是1
# number of vessel types
num_classes = 4




"""
Build the classifier=========================
"""
# 输入层
input_layer=layers.Input(shape=input_shape)

# 卷积层堆叠
x = layers.Conv2D(1, (10, 10), activation='relu', strides=1, padding='same')(input_layer)
x = layers.Conv2D(1, (10, 10), activation='relu', strides=1, padding='same')(x)
x = layers.Conv2D(5, (5, 5), activation='relu', strides=1, padding='same')(x)
x = layers.Conv2D(5, (5, 5), activation='relu', strides=1, padding='same')(x)
x = layers.Conv2D(3, (3, 3), activation='relu', strides=1, padding='same')(x)

# 扁平化后接全连接层
x = layers.Flatten()(x)
x = layers.Dense(250, activation='relu')(x)

# 输出层
output_layer = layers.Dense(num_classes, activation='softmax')(x)  # num_classes是输出的分类数，根据实际情况替换

# 创建模型
classifier = models.Model(inputs=input_layer, outputs=output_layer)

# 模型概要
classifier.summary()
"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""



"""
Build the encoder============================
"""


# 输入层：假设我们已经有了处理好的轨迹数据和标签
input_shape = [height, width, 1]  # 轨迹数据形状
label_shape = (num_classes,)  # 标签的形状，使用one-hot编码

# 轨迹输入
x_input = layers.Input(shape=input_shape, name='input_trajectory')

# 标签输入
label_input = layers.Input(shape=label_shape, name='input_label')

# 卷积块
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x_input)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
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
encoder = models.Model(inputs=[x_input, label_input], outputs=[z_mean, z_log_var, z], name='encoder')

# 模型概要
encoder.summary()
"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""



"""
Build the decoder============================
"""
# 假设潜在变量z的维度和标签y的one-hot编码维度已知
latent_dim = 10 # 潜在空间的维度
num_classes = 4 # 分类的数量，即one-hot编码的维度

# 潜在变量和标签的输入层
z_input = layers.Input(shape=(latent_dim,), name='input_latent')
label_input = layers.Input(shape=(num_classes,), name='input_label')

# 特征拼接
concatenated_features = layers.concatenate([z_input, label_input])

# 通过两个全连接层生成特征向量
x = layers.Dense(250, activation='relu')(concatenated_features)
x = layers.Dense(250, activation='relu')(x)

# 将特征向量恢复到合适的维度以便进行上采样
x = layers.Reshape((5, 5, 10))(x) # 这里的目标维度取决于您想要如何开始上采样过程

# 五个反卷积层（上采样层），使用给定的输出通道数和内核大小
x = layers.Conv2DTranspose(5, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(5, kernel_size=(5, 5), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(5, kernel_size=(10, 10), activation='relu', padding='same')(x)
x = layers.Conv2DTranspose(5, kernel_size=(10, 10), activation='relu', padding='same')(x)

# 最后一个反卷积层生成原始轨迹，使用sigmoid激活函数
x_output = layers.Conv2DTranspose(1, kernel_size=(10, 10), activation='sigmoid', padding='same')(x)

# 创建解码器模型
decoder = models.Model(inputs=[z_input, label_input], outputs=x_output, name='decoder')

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
        keras.losses.binary_crossentropy(x, reconstruction), axis=(1, 2)
    )
    
    # Compute the KL divergence loss
    kl_loss = -0.5 * tf.reduce_mean(
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1
    )
    
    # Sum the two loss terms
    loss = reconstruction_loss + kl_loss
    return loss
def compute_loss_unlabeled(encoder, decoder, classifier, x):
    # Sum over all possible labels
    u_loss = 0
    all_possible_labels=[0,1,2,3]

    predict_res=classifier(x)
    ## 船只种类
    for y in all_possible_labels:  # all_possible_labels should be one-hot encoded labels
        label = tf.one_hot(indices=y, depth=num_classes)
        loss = compute_loss_labeled(encoder, decoder, x, label)
        q_y_given_x=predict_res[:,y]
        u_loss += q_y_given_x * (-loss)  # q_y_given_x should be the predicted probability of label y
          
    # Add the entropy of the predicted label distribution
    q_y_given_x = classifier(x)
    entropy_loss = - tf.reduce_mean(tf.reduce_sum(q_y_given_x * tf.math.log(q_y_given_x + 1e-7), axis=1))
    
    # Sum the negative lower bound and entropy loss
    total_u_loss = u_loss + entropy_loss
    return total_u_loss
def compute_classifier_loss(classifier, x_labeled, y_true):
    # Classifier predictions
    y_pred = classifier(x_labeled)
    
    # Compute the cross-entropy loss
    clf_loss = tf.reduce_mean(
        keras.losses.categorical_crossentropy(y_true, y_pred)

    )
    
    return clf_loss




"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
class VAE(keras.Model):

    def __init__(self, encoder,decoder,clf,**kwargs):
        # 调用父类的__init__()
        super(VAE,self).__init__(**kwargs)
        self.encoder=encoder
        self.decoder=decoder
        self.clf=clf
# train step每次会自动调取数据集 
    def train_step(self,data):
        x,target_x=data

        # 对于无标签的AIS数据，traget_x应该全0，有标签的则是one-hot形式，存在一个1。
        # 判断是否是标记数据

        mt_label=False
        mt_unlabel=False

        n1=0
        n2=0

        label_x=np.empty(x[0].shape)
        unlabel_x=np.empty(x[0].shape)

        label_target_x=np.empty(target_x[0].shape)
        unlabel_target_x=np.empty(target_x[0].shape)


        for i in range(len(target_x)):
            if target_x[i]>=0:
                if mt_label==False:
                    mt_label=True
                    label_x=x[i]
                    label_target_x=target_x[i]
                else:
                    label_x=np.vstack((label_x,x[i]))
                    label_target_x=np.vstack((label_target_x,target_x[i]))
                n1=n1+1
            elif target_x[i]==-1:
                if mt_unlabel==False:
                    mt_unlabel=True
                    unlabel_x=x[i]
                    unlabel_target_x=target_x[i]
                else:
                    unlabel_x=np.vstack((unlabel_x,x[i]))
                    unlabel_target_x=np.vstack((unlabel_target_x,target_x[i]))
                n2=n2+1



        alpha=beta*float(n2/n1)
        with tf.GradientTape as Tape:
            # 均值、方差、z

            Loss_1=compute_loss_labeled(self.encoder,self.decoder,label_x,label_target_x)
            Loss_2=compute_loss_unlabeled(self.encoder,self.decoder,self.clf,unlabel_x)
            Loss_clf=compute_classifier_loss(self.clf,x_labeled=label_x,y_true=label_target_x)
            Loss=Loss_1+Loss_2+alpha*Loss_clf
        grads = Tape.gradient(Loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            "loss": Loss,
            "Loss_1": Loss_1,
            "Loss_2": Loss_2,
            "Loss_clf": Loss_clf,
            "alpha": alpha,
        }
# (x_train, target_train), (x_test, target_test) = keras.datasets.mnist.load_data()
