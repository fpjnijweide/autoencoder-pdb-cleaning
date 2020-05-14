import os
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, \
    Reshape, BatchNormalization, Dropout, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l1, l2, l1_l2
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32')/255
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_val = x_train[5000:6000]
x_train = x_train[:5000]


encoding_dim = 32
input_img = Input(shape=(28, 28, 1))

x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(32, activation='relu')(x)


encoded = Dense(encoding_dim, activation='relu', activity_regularizer=l1(0.001), name='encoded')(x)


# , activity_regularizer=l1(0.01)

x = Dense(32, activation='relu')(encoded)
x = Dense(7*7*64, activation='relu')(x)
x = Reshape((7, 7, 64))(x)
x = Conv2DTranspose(64, (3, 3), (2, 2), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
# x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2DTranspose(32, (3, 3), (2, 2), activation='relu', padding='same')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

hist = History()

autoencoder.compile(optimizer='adam', loss='mse')

for e in range(3):
    start_time = time.time()

    autoencoder.fit(x_train, x_train, epochs=3, batch_size=256, shuffle=True, validation_data=(x_val, x_val), callbacks=[hist])
    print("--- %s seconds ---" % (time.time() - start_time))
    autoencoder.save('model.h5')

    #print(str(hist.history.items()))

    #with open('testlog.txt', 'a') as f:
        #f.write('losses for epoch {}, '.format(3) + str(hist.history.items()) + '\n')
# for elem in vectorstudy:
#     print(len(np.argwhere(elem).tolist()))


# first100imgp = autoencoder.predict(x_train[0:100])
# first100imgp = first100imgp*255
# first100imgp = first100imgp.astype('uint8')
# first100imgp = first100imgp.reshape(100, 28, 28)
#
# first100img = x_train[0:100]
# first100img = first100img*255
# first100img = first100img.astype('uint8')
# first100img = first100img.reshape(100, 28, 28)
#
# for i in range(10):
#     image1 = first100img[i, :].squeeze()
#     plt.subplot(10, 2, 2*i+1)
#     plt.imshow(image1, cmap='gray')
#     image2 = first100imgp[i, :].squeeze()
#     plt.subplot(10, 2, 2*i+2)
#     plt.imshow(image2, cmap='gray')
#
# # plt.show()
#
#
# encoderfeatures = encoder.predict(x_train[:500])
# label = y_train[:500]
#
# tsne = TSNE(n_components=2, verbose=1, perplexity=23, n_iter=100000, learning_rate=200)
# pca = PCA(n_components=2)
# kmeans = KMeans(n_clusters=10, n_init=30)
#
# x_tsne = tsne.fit_transform(encoderfeatures)
#
# x = x_tsne[:, 0]
# y = x_tsne[:, 1]
#
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.5', '0.3', '0.7']
#
# fig = plt.figure()
# for i in range(500):
#     plt.scatter(x[i], y[i], c=colors[label[i]], label=label[i] if str(label[i])
#                 not in plt.gca().get_legend_handles_labels()[1] else '')
#
# fig.legend()
# plt.show()

