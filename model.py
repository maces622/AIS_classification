import os

os.environ["KERAS_BACKEND"] = "tensorflow"
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from tensorflow import train
from keras.callbacks import ModelCheckpoint
import pickle as pkl
import matplotlib.pyplot as plt
import math
from keras.regularizers import l1, l2,l1_l2
import pandas as pd
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
latent_dim = 6
# shape : 输入形状 长 宽 通道数目
# AIS数据重新采样后长度为 180（暂定），字段数目为 10 ，独热码形式通道为1(gray graph 1,RGB 3)
height = 160
width  = 400
chan = 1
beta=1000
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
x = layers.Conv2D(1, (3, 3), activation='relu', strides=1, padding='same')(x)

# 扁平化后接全连接层
x = layers.Flatten()(x)
# x = layers.Dense(250, activation='relu')(x)
x = layers.Dense(250, activation='relu',kernel_regularizer=l1_l2(l1=0.05,l2=0.05))(x)

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
x = layers.Conv2D(1, (3, 3), activation='relu', strides=1, padding='same')(x)

x = layers.Flatten()(x)

# 全连接层获取轨迹特征
# x_features = layers.Dense(250, activation='relu')(x)

x_features = layers.Dense(250, activation='relu',kernel_regularizer=l2(0.05))(x)

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
x = layers.Dense(1250, activation='relu')(concatenated_features)

# x = layers.Dense(1250, activation='relu',)(x)

x = layers.Dense(1250, activation='relu',kernel_regularizer=l2(0.05))(x)

# 将特征向量恢复到合适的维度以便进行上采样
x = layers.Reshape((10, 25, 5))(x) # 这里的目标维度取决于您想要如何开始上采样过程

# 五个反卷积层（上采样层），使用给定的输出通道数和内核大小
x = layers.Conv2DTranspose(1, kernel_size=(3, 3), activation='relu', strides=2,padding='same')(x)
x = layers.Conv2DTranspose(5, kernel_size=(5, 5), activation='relu',  strides=2,padding='same')(x)
x = layers.Conv2DTranspose(5, kernel_size=(10, 10), activation='relu',  strides=2,padding='same')(x)
x = layers.Conv2DTranspose(5, kernel_size=(10, 10), activation='relu',  strides=2,padding='same')(x)
x_output = layers.Conv2DTranspose(1, kernel_size=(10, 10), activation='sigmoid',  strides=1,padding='same')(x)

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
@keras.utils.register_keras_serializable()
def compute_loss_labeled(encoder, decoder, x, y):
    # Encode the input to get the mean, log-variance, and sampled z
    z_mean, z_log_var, z = encoder([x, y])
    # Decode the sampled z to reconstruct the image
    reconstruction = decoder([z, y])

    # Compute the binary cross-entropy for reconstruction loss
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(keras.losses.binary_crossentropy(x,reconstruction),
                      axis=(1,2))
    )

    # Compute the KL divergence loss
    kl_loss = -0.5 * (
        1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss,axis=1))
    # Sum the two loss terms
    loss = reconstruction_loss + kl_loss
    return loss,kl_loss
batch_size=100
@keras.utils.register_keras_serializable()
def compute_loss_unlabeled(encoder, decoder, classifier, x):
    # Sum over all possible labels
    u_loss = 0.0
    all_possible_labels=[0,1,2,3]

    predict_res=classifier(x)
    # tf.print(predict_res)
    ## 船只种类
    for y in all_possible_labels:  # all_possible_labels should be one-hot encoded labels
        arrays = np.full((25,), y)
        label_1 = np.eye(4)[arrays]
        label_1 = np.expand_dims(label_1, -1)
        y_tensor = tf.convert_to_tensor(label_1)

        loss,kl_loss = compute_loss_labeled(encoder, decoder, x, y_tensor)
        q_y_given_x=predict_res[:,y]
        u_loss += q_y_given_x * (loss)  # q_y_given_x should be the predicted probability of label y

    # Add the entropy of the predicted label distribution
    entropy_loss = tf.reduce_sum(predict_res * tf.math.log(predict_res + tf.keras.backend.epsilon()), axis=-1)
    # tf.print(entropy_loss)
    # Sum the negative lower bound and entropy loss
    total_u_loss = u_loss + entropy_loss
    return total_u_loss
@keras.utils.register_keras_serializable()
def compute_classifier_loss(classifier, x_labeled, y_true):
    # Classifier predictions
    y_pred = classifier(x_labeled)
    # Compute the cross-entropy loss
    y_true_squeezed = tf.squeeze(y_true, axis=-1)
    class_weights = tf.constant([1/1,1/5, 1/3, 1/1])  # 根据实际类别的不平衡程度调整
    # class_weights = tf.constant([1.0,1.0, 1.0, 1.0])  # 根据实际类别的不平衡程度调整
    loss = keras.losses.categorical_crossentropy(y_true_squeezed, y_pred)
    sample_weights = tf.reduce_sum(class_weights * y_true_squeezed, axis=1)  # 应用权重
    weighted_loss = tf.reduce_mean(loss * sample_weights)
   
    return weighted_loss




"""
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

@keras.utils.register_keras_serializable()
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
    
    def call(self, inputs):
        x_supervised, y_supervised = inputs
        z_mean, z_log_var, z = self.encoder([x_supervised, y_supervised])
        reconstructed = self.decoder([z, y_supervised])
        return reconstructed
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
        [x_supervised, x_unsupervised],y_supervised = data
        # print(x_supervised.shape,y_supervised.shape,x_unsupervised.shape)
        # x_unsupervised=x_supervised
        alpha=beta*float(0.5)
        half_u=x_unsupervised[:25,:,:,:]
        # print(type(x_unsupervised))
        # print(x_unsupervised.shape)
        with tf.GradientTape() as Tape:
            # 均值、方差、z

            Loss_1,kl_loss=compute_loss_labeled(self.encoder,self.decoder,x_supervised,y_supervised)
            # print(type(x_unsupervised))

            Loss_2=compute_loss_unlabeled(self.encoder,self.decoder,self.clf,half_u)
            Loss_clf=compute_classifier_loss(self.clf,x_labeled=x_supervised,y_true=y_supervised)
            Loss=(Loss_1+Loss_2*0.25)+alpha*Loss_clf

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
@keras.utils.register_keras_serializable()
def convert_to_one_hot(data):
    # 获取数据的形状
    n, _, channels = data.shape
    # 初始化一个用于存储结果的数组
    total_bins=400
    result = np.zeros((n, 160, total_bins))

    # 遍历每一个数据样本
    for i in range(n):
        # 遍历每一个通道
        channel_4_data=data[i,:,4]
        channel_5_data=data[i,:,5]
        channel_6_data=data[i,:,6]
        for j in range(4):
            channel_data = data[i, :, j]
            one_hot=np.eye(50)[np.round(channel_data*49).astype(int)]
            result[i,:,j*50:(j+1)*50]=one_hot
        # for j in range(2):
        #     channel_data = data[i, :, j+2]
        #     one_hot=np.eye(10)[np.round(channel_data*9).astype(int)]
        #     result[i,:,(j+2)*10:(j+2+1)*10]=one_hot

        one_hot_4=np.eye(75)[np.round(channel_4_data*74).astype(int)]
        one_hot_5=np.eye(75)[np.round(channel_5_data*74).astype(int)]
        one_hot_6=np.eye(50)[np.round(channel_6_data*49).astype(int)]
        result[i, :, 40*5:40*5+75] = one_hot_4
        result[i, :, 275:275+75] = one_hot_5
        result[i, :, 350:350+50] = one_hot_6


    return result

with open('procd_california.pkl','rb') as f:
    data = pkl.load(f)


"""
将数据集分为训练集和测试集，
并将AIS字段和其船只类型标签拆开
"""

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
        min_drt=min(min_drt,row[8])

data[:,:,2]=(data[:,:,2]-min_lat)/(max_lat-min_lat)
data[:,:,3]=(data[:,:,3]-min_lon)/(max_lon-min_lon)
data[:,:,4]=(data[:,:,4]-min_sog)/(max_sog-min_sog)
data[:,:,5]=(data[:,:,5]-min_cog)/(max_cog-min_cog)

data[:,:,6]=(data[:,:,6]-min_len)/(max_len-min_len)
data[:,:,7]=(data[:,:,7]-min_wid)/(max_wid-min_wid)
data[:,:,8]=(data[:,:,8]-min_drt)/(max_drt-min_drt)
print(max_lat,min_lat)
print(max_lon,min_lon)
print(max_sog,min_sog)
print(max_cog,min_cog)

print(max_len,min_len)
print(max_wid,min_wid)
print(max_drt,min_drt)
np.random.seed(42)

np.random.shuffle(data)
print(data.shape[0])
print("000000000000000000000000000---------")
_train=data[:2000].copy()
_test=data[2000:2500].copy()
_test=np.concatenate((_test,_test,_test,_test),axis=0)
print(type(_test))
_res=data[2500:3700].copy()

x_train=_train[:,:,[2,3,4,5,6,7,8]].astype(float)
target_train=_train[:,0,9].astype(int)

x_test=_test[:,:,[2,3,4,5,6,7,8]].astype(float)
target_test=_test[:,0,9].astype(int)

x_res=_res[:,:,[2,3,4,5,6,7,8]].astype(float)
# print(_test[0][0])
target_res=_res[:,0,9].astype(int)
target_mmsi=_res[:,0,1].astype(int)
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


conditions3 = [
    target_res == 3,
    target_res == 6,
    target_res == 7,
    target_res == 8,
]

# 定义对应条件下的替换值
choices = [0, 1, 2, 3]

# 应用 numpy.select

arr_new = np.select(conditions1, choices, default=target_train)
target_train_1=np.eye(4)[arr_new]
arr_new1 = np.select(conditions2, choices, default=target_test)
target_test_1=np.eye(4)[arr_new1]
arr_new2 = np.select(conditions3, choices, default=target_res)
target_res_1=np.eye(4)[arr_new2]

x_train=convert_to_one_hot(x_train)
x_test=convert_to_one_hot(x_test)
x_res=convert_to_one_hot(x_res)

x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)
x_res =  np.expand_dims(x_res, -1)
target_train_1 = np.expand_dims(target_train_1,-1)
target_res_1=np.expand_dims(target_res_1,-1)

# print(x_train.shape)
epoch_losses = []

def on_epoch_end(epoch, logs):
    epoch_losses.append(logs['Loss_clf'])

# 将回调函数传递给训练过程
callbacks = [tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)]
encoder.load_weights('enc_1.h5')
classifier.load_weights('clf_1.h5')
vae = VAE(encoder, decoder,classifier)
vae.compile(optimizer=keras.optimizers.Adam())

# vae.load_weights('vae_weights.h5')
# vae.fit([x_train,x_test],target_train_1, epochs=25, batch_size=batch_size,callbacks=callbacks)
# vae.save_weights('vae_weights.h5')




loss_lst=[]

# 输出每个epoch的平均损失
for i, loss in enumerate(epoch_losses):
    print(f"Epoch {i+1}: Average CLF Loss = {loss}")
    loss_lst.append(loss)
# with open(f"loss_beta_is{beta}_1.pkl","wb") as f:
#     pkl.dump(loss_lst,f)

group3=[]
group6=[]
group7=[]
group8=[]
record_key_v=[]
confusion_matrix = np.zeros((4, 4),dtype=int)
file_path = 'mmsi_vessel_type.csv'
columns=["MMSI","class"]
# 判断文件是否存在
if os.path.exists(file_path):
    df1 = pd.read_csv('mmsi_vessel_type.csv')

else:
    df1 = pd.DataFrame(columns=columns)
# 使用append添加到df1
df1.at[0, 'MMSI'] = 333333333
df1.at[0, 'class'] = 'Passenger'
# print(df1)

def plot_label_clusters(encd, clfi, data, y):
    global df1
    # Display a 2D plot of the digit classes in the latent space
    cnt = 0
    z_mean, _, _ = encd.predict([data,y])
    clf=clfi.predict(data)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')  # 创建一个三维的绘图工具
    ax.scatter(z_mean[:190, 0], z_mean[:190, 1], z_mean[:190, 1], c=arr_new2[:190])
    # plt.colorbar()
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_zlabel("z[2]")
    plt.show()
    idx=0
    for x in range(clf.shape[0]):
        # print(clf[x],arr_new2[x])
        mx=0
        clp=-1
        for g in range(4):
            if clf[x][g]>mx:
                mx=clf[x][g]
                clp=g
        if clp==arr_new2[x]:
            cnt=cnt+1
        new_data ={'MMSI': target_mmsi[x], 'class': arr_new2[x]}

        df1.loc[len(df1)] = new_data
        confusion_matrix[arr_new2[x]][clp]+=1
        # df = df.append(new_data, ignore_index=True)

        if clp==0:
            group3.append(_res[idx])
        elif clp==1:
            group6.append(_res[idx])
        elif clp==2:
            group7.append(_res[idx])
        elif clp==3:
            group8.append(_res[idx])
        idx+=1

    print("测试集数量：",clf.shape[0])
    print("准确数量",cnt)


plot_label_clusters(encoder, classifier, x_res, target_res_1)
# df1.to_csv('mmsi_vessel_type.csv', index=False)
unique_class_counts = df1.groupby('MMSI')['class'].nunique()
mmsis_with_multiple_classes = unique_class_counts[unique_class_counts > 1].index
entries_with_multiple_classes = df1[df1['MMSI'].isin(mmsis_with_multiple_classes)]
entries_with_multiple_classes
print(entries_with_multiple_classes)
print(len(entries_with_multiple_classes))

# check C



#
# np_group3=np.stack(group3,axis=0)
# np_group6=np.stack(group6,axis=0)
# np_group7=np.stack(group7,axis=0)
# np_group8=np.stack(group8,axis=0)
# with open("group3.pkl",'wb') as f:
#     pkl.dump(np_group3,f)
#
# with open("group6.pkl",'wb') as f:
#     pkl.dump(np_group6,f)
#
# with open("group7.pkl",'wb') as f:
#     pkl.dump(np_group7,f)
#
# with open("group8.pkl",'wb') as f:
#     pkl.dump(np_group8,f)

print(confusion_matrix)
"""
400:400
测试集数量： 250
准确数量 167

600:300
测试集数量： 150
准确数量 114

600:300
测试集数量： 150
准确数量 114

测试集数量： 200
准确数量 164
"""

"""
1101
[[ 72  18   0   0]
 [ 17 552   3   3]
 [  1  15 342  17]
 [  0   5  20 135]]
 
[[ 72  16   0   2]
 [  9 563   1   2]
 [  3   6 354  12]
 [  0   2  15 143]]


[[ 69  16   2   3]
 [ 13 554   4   4]
 [  0  11 325  39]
 [  0   1  15 144]]
 
 m1
 1108
[[ 67  20   2   1]
 [ 17 555   3   0]
 [  1  15 343  16]
 [  0   3  21 136]]
[[ 62  25   2   1]
 [ 14 557   4   0]
 [  2   8 349  16]
 [  0   1  19 140]]
1118
[[ 77  12   1   0]
 [ 27 541   6   1]
 [  3   1 358  13]
 [  0   1  17 142]]
m2
1110
[[ 77  11   0   2]
 [ 23 549   2   1]
 [  2   8 348  17]
 [  0   1  23 136]]

[[ 78   7   4   1]
 [ 19 548   5   3]
 [  3   5 312  55]
 [  0   0   4 156]]
"""
