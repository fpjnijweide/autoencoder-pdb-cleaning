'''
Copyright (c) 2018 Norio Tamada
Released under the MIT license
https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/lib/keras_ex/gkernel/LICENSE.md
'''

# This file was edited by F.P.J. Nijweide to use tensorflow.keras instead of keras and RandomUniform instead of random_uniform
# I do not claim responsibility for the lines below, they are the result of the work by Norio Tamada

import numpy as np
from tensorflow.keras import initializers, constraints
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

class GaussianKernel(Layer):
    
    def __init__(self, num_landmark, num_feature,
                 kernel_initializer='glorot_uniform',
                 kernel_constraint=None,
                 kernel_gamma='auto',
                 **kwargs):
        '''
        num_landmark:
            number of landmark
            that was number of output features
        num_feature:
            depth of landmark
            equal to inputs.shape[1]
        kernel_gamma:
            kernel parameter
            if 'auto', use 1/(2 * d_mean**2)
            d is distance between samples and landmark
            d_mean is mean of d
        '''
        super(GaussianKernel, self).__init__(**kwargs)
        
        self.output_dim = num_landmark
        self.num_feature = num_feature
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_constraint = constraints.get(kernel_constraint)

        # kernel parameter
        self.kernel_gamma = kernel_gamma

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.num_feature),
                                      initializer=self.kernel_initializer,
                                      constraint=self.kernel_constraint)
        super(GaussianKernel, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, training=None):
        return self.gauss(x, self.kernel, self.kernel_gamma)
    
    def gauss(self, x, landmarks, gamma):
        x2 = K.sum(K.square(x), axis=1)
        x2 = K.reshape(x2, (-1,1))
        x2 = K.repeat_elements(x2, self.output_dim, axis=1)
        lm2 = K.sum(K.square(landmarks), axis=1)
        xlm = K.dot(x, K.transpose(landmarks))
        
        ret = x2 + lm2 - 2*xlm
        if gamma == 'auto':
            '''
            gamma is calculated by each batch
            '''
            d = K.sqrt(ret)
            d_mean = K.mean(d)
            gamma = 1. / (2. * d_mean**2)
        ret = K.exp(-gamma * ret)
        return ret
    
    def get_config(self):
        config = {
            'num_landmark': self.output_dim,
            'num_feature': self.num_feature,
            'kernel_gamma': self.kernel_gamma,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
        }
        base_config = super(GaussianKernel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianKernel2(Layer):
    
    def __init__(self, landmarks, **kwargs):
        '''
        landmarks:
            fixed landmarks using
        '''
        super(GaussianKernel2, self).__init__(**kwargs)
        if isinstance(landmarks, (list,)):
            landmarks = np.array(landmarks)
        self.landmarks = landmarks.astype(np.float32)
        self.num_landmark, self.num_feature = landmarks.shape
        self.output_dim = self.num_landmark
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.gamma_elm = self.add_weight(name='gamma_elm',
                                      shape=(1, ),
                                      initializer=initializers.RandomUniform(-2, -1))
        super(GaussianKernel2, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, training=None):
        return self.gauss(x, self.landmarks, K.exp(self.gamma_elm), training=training)
    
    def gauss(self, x, landmarks, gamma, training=None):
        x2 = K.sum(K.square(x), axis=1)
        x2 = K.reshape(x2, (-1,1))
        x2 = K.repeat_elements(x2, self.output_dim, axis=1)
        lm2 = K.sum(K.square(landmarks), axis=1)
        xlm = K.dot(x, K.transpose(landmarks))
        ret = x2 + lm2 - 2*xlm
        ret = K.exp(-gamma * ret)
        return ret
    
    def get_config(self):
        config = {
            'landmarks': self.landmarks,
        }
        base_config = super(GaussianKernel2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GaussianKernel3(Layer):
    
    def __init__(self, num_landmark, num_feature,
                 kernel_initializer='glorot_uniform',
                 **kwargs):
        '''
        num_landmark:
            number of landmark
            that was number of output features
        num_feature:
            depth of landmark
            equal to inputs.shape[1]
        '''
        super(GaussianKernel3, self).__init__(**kwargs)
        
        self.output_dim = num_landmark
        self.num_feature = num_feature
        self.kernel_initializer = initializers.get(kernel_initializer)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
    
    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.num_feature),
                                      initializer=self.kernel_initializer)
        self.gamma_elm = self.add_weight(name='gamma_elm',
                                      shape=(1, ),
                                      initializer=initializers.RandomUniform(-2, -1))
        super(GaussianKernel3, self).build(input_shape)  # Be sure to call this somewhere!
    
    def call(self, x, training=None):
        return self.gauss(x, self.kernel, K.exp(self.gamma_elm))
    
    def gauss(self, x, landmarks, gamma, training=None):
        x2 = K.sum(K.square(x), axis=1)
        x2 = K.reshape(x2, (-1,1))
        x2 = K.repeat_elements(x2, self.output_dim, axis=1)
        lm2 = K.sum(K.square(landmarks), axis=1)
        xlm = K.dot(x, K.transpose(landmarks))
        ret = x2 + lm2 - 2*xlm
        ret = K.exp(-gamma * ret)
        return ret
    
    def get_config(self):
        config = {
            'num_landmark': self.output_dim,
            'num_feature': self.num_feature,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
        }
        base_config = super(GaussianKernel3, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))