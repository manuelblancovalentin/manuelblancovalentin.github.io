"""
"""
__author__ = "Manuel Blanco Valentin"

""" Basic modules """
import numpy as np

""" TF """
import tensorflow as tf
import tensorflow.keras.backend as K

# ========================================================================================
# Little snippet to apply colormap to tensor
# ========================================================================================
def colmap(x, map="bwr"):
    vmin = tf.reduce_min(x)
    vmax = tf.reduce_max(x)
    x = (x - vmin)/(vmax-vmin)
    R = tf.clip_by_value(-2*(x-1),0.,1.)
    B = tf.clip_by_value(2*x,0.,1.)
    G = tf.minimum(R,B)

    y = tf.concat((R,G,B),2)
    return y

def noise(n, latent_size):
    return np.random.normal(0.0, 1.0, size = [n, latent_size]).astype('float32')

def noiseList(n, n_layers, latent_size):
    return [noise(n, latent_size)] * n_layers

def mixedList(n, n_layers, latent_size):
    tt = int(np.random.random() * n_layers)
    p1 = [noise(n, latent_size)] * tt
    p2 = [noise(n, latent_size)] * (n_layers - tt)
    return p1 + [] + p2

def noise_image(n, im_shape):
    return np.random.uniform(0.0, 1.0, size = (n,) + im_shape).astype('float32')


#Lambdas
def crop_to_fit(x):
    height = x[1].shape[1]
    width = x[1].shape[2]
    return x[0][:, :height, :width, :]

def upsample(x):
    return K.resize_images(x,2,2,"channels_last",interpolation='bilinear')

def upsample_to_size(x, im_size):
    y = im_size / x.shape[2]
    x = K.resize_images(x, y, y, "channels_last",interpolation='bilinear')
    return x
