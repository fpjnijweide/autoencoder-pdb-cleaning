from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np

test_corr = 0
structure = 0
train_corr = 0

encoding_dim = 3
numb_cat = len(structure)
inputs = Input(shape=(train_corr.shape[1],))
encoded = Dense(6, activation='linear')(inputs)
encoded = Dense(encoding_dim, activation='linear')(encoded)
decoded = Dense(6, activation='linear')(encoded)
decodes = [Dense(e, activation='softmax')(decoded) for e in structure]

losses = [jsd for j in range(numb_cat)]  # JSD loss function
autoencoder = Model(inputs, decodes)
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
autoencoder.compile(optimizer=sgd, loss=losses, loss_weights=[1 for k in range(numb_cat)])

train_attr_corr = [train_corr[:, i:j] for i, j in zip(np.cumsum(structure_0[:-1]), np.cumsum(structure_0[1:]))]
test_attr_corr = [test_corr[:, i:j] for i, j in zip(np.cumsum(structure_0[:-1]), np.cumsum(structure_0[1:]))]

history = autoencoder.fit(train_corr, train_attr_corr, epochs=100, batch_size=2,
                          shuffle=True,
                          verbose=1, validation_data=(test_corr, test_attr_corr))
