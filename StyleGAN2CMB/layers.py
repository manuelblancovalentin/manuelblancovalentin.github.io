# -*- coding: utf-8 -*-
"""Convolutional layers.
"""

from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import conv_utils
import tensorflow as tf

""" Deep custom layers & utils """
from .utils import upsample_to_size, crop_to_fit, upsample

""" 
Blocks 
"""

""" Generator block """
def g_block(inp, istyle, inoise, fil, nchannels, im_size, u = True):

    if u:
        #Custom upsampling because of clone_model issue
        out = tf.keras.layers.Lambda(upsample, output_shape=[None, inp.shape[2] * 2, inp.shape[2] * 2, None])(inp)
    else:
        out = tf.keras.layers.Activation('linear')(inp)

    rgb_style = tf.keras.layers.Dense(fil, kernel_initializer = tf.keras.initializers.VarianceScaling(200/out.shape[2]))(istyle)
    style = tf.keras.layers.Dense(inp.shape[-1], kernel_initializer = 'he_uniform')(istyle)
    delta = tf.keras.layers.Lambda(crop_to_fit)([inoise, out])
    d = tf.keras.layers.Dense(fil, kernel_initializer = 'zeros')(delta)

    out = Conv2DMod(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')([out, style])
    out = tf.keras.layers.add([out, d])
    out = tf.keras.layers.LeakyReLU(0.2)(out)

    style = tf.keras.layers.Dense(fil, kernel_initializer = 'he_uniform')(istyle)
    d = tf.keras.layers.Dense(fil, kernel_initializer = 'zeros')(delta)
    out = Conv2DMod(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')([out, style])
    out = tf.keras.layers.add([out, d])
    out = tf.keras.layers.LeakyReLU(0.2)(out)
    return out, to_rgb(out, rgb_style, nchannels, im_size)

""" Discriminator block """
def d_block(inp, fil, p = True):
    res = tf.keras.layers.Conv2D(fil, 1, kernel_initializer = 'he_uniform')(inp)
    out = tf.keras.layers.Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(inp)
    out = tf.keras.layers.LeakyReLU(0.2)(out)
    out = tf.keras.layers.Conv2D(filters = fil, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform')(out)
    out = tf.keras.layers.LeakyReLU(0.2)(out)
    out = tf.keras.layers.add([res, out])
    if p:
        out = tf.keras.layers.AveragePooling2D()(out)
    return out

def to_rgb(inp, style, nchannels, im_size):
    size = inp.shape[2]
    x = Conv2DMod(nchannels, 1, kernel_initializer = tf.keras.initializers.VarianceScaling(200/size), demod = False)([inp, style])
    return tf.keras.layers.Lambda(lambda x, im_size=im_size: upsample_to_size(x, im_size),
                                  output_shape=[None, im_size, im_size, None])(x)

def from_rgb(inp, im_size, conc = None):
    fil = int(im_size * 4 / inp.shape[2])
    z = tf.keras.layers.AveragePooling2D()(inp)
    x = tf.keras.layers.Conv2D(fil, 1, kernel_initializer = 'he_uniform')(z)
    if conc is not None:
        x = tf.keras.layers.Concatenate()([x, conc])
    return x, z


""" Custom Conv2DMod layer """
class Conv2DMod(tf.keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 demod=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.demod = demod
        self.input_spec = [tf.keras.layers.InputSpec(ndim = 4),
                            tf.keras.layers.InputSpec(ndim = 2)]

    def build(self, input_shape):
        #print('input_shape',input_shape)
        channel_axis = -1
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[0][channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        if input_shape[1][-1] != input_dim:
            raise ValueError('The last dimension of modulation input should be equal to input dimension.')
        #print('kernel_shape',kernel_shape)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        # Set input spec.
        self.input_spec = [tf.keras.layers.InputSpec(ndim=4, axes={channel_axis: input_dim}),
                            tf.keras.layers.InputSpec(ndim=2)]
        self.built = True

    def call(self, inputs, **kwargs):

        #To channels last
        #x = tf.transpose(inputs[0], [0, 3, 1, 2])
        x = inputs[0]
        #print('1',x)

        #Get weight and bias modulations
        #Make sure w's shape is compatible with self.kernel
        w = K.expand_dims(K.expand_dims(K.expand_dims(inputs[1][-1], axis = 0), axis = 0), axis = -1)
        #print('2',w)
        #Add minibatch layer to weights
        #wo = K.expand_dims(self.kernel, axis = 0)
        wo = self.kernel

        #Modulate
        weights = wo * (w+1)
        #print('3',wo,self.kernel,weights)

        #Demodulate
        if self.demod:
            d = K.sqrt(K.sum(K.square(weights), axis=[1,2,3], keepdims = True) + 1e-8)
            weights = weights / d

        #Reshape/scale input
        #x = tf.reshape(x, [1, x.shape[1], x.shape[2], -1]) # Fused => reshape minibatch to convolution groups.
        #w = tf.reshape(tf.transpose(weights, [1, 2, 3, 0, 4]), [weights.shape[1], weights.shape[2], weights.shape[3], -1])
        w = weights
        #print('4',w,x)
        #x = tf.transpose(x, [0, 2, 3, 1])
        #print('5',w, x)
        x = tf.nn.conv2d(x, w,
                strides=self.strides,
                padding="SAME",
                data_format="NHWC")
        #print('6',x)
        # Reshape/scale output.
        #x = tf.reshape(x, [-1, x.shape[1], x.shape[2], self.filters]) # Fused => reshape convolution groups back to minibatch.
        #print('7',x)
        #print(inputs[0], x)
        return x

    def compute_output_shape(self, input_shape):
        space = input_shape[0][1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.filters,)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'activity_regularizer':
                tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'demod': self.demod
        }
        base_config = super(Conv2DMod, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
