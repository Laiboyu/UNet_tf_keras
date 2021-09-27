from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten, Concatenate
from keras import models, optimizers
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras import layers
from keras.models import load_model
import keras
import tensorflow as tf
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
from keras.utils.vis_utils import plot_model
import numpy as np


    
########################### Base construction ##############################
def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = MaxPooling2D((2, 2), (2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c
################################# Unet #####################################

def UNet(image_size = (512, 512, 1)):
    f = [16, 32, 64, 128, 256, 512]
    inputs = Input(image_size)
#     inputs = data_augmentation(inputs)
    
    # downsampling 
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) # 512 -> 256
    c2, p2 = down_block(p1, f[1]) # 256 -> 128
    c3, p3 = down_block(p2, f[2]) # 128 -> 64
    c4, p4 = down_block(p3, f[3]) # 64 -> 32
    c5, p5 = down_block(p4, f[4]) # 32 -> 16
    
    bn = bottleneck(p5, f[5])
    
    # upsampling
    u1 = up_block(bn, c5, f[4]) #16 -> 32
    u2 = up_block(u1, c4, f[3]) #32 -> 64
    u3 = up_block(u2, c3, f[2]) #64 -> 128
    u4 = up_block(u3, c2, f[1]) #128 -> 256
    u5 = up_block(u4, c1, f[0]) #256 -> 512
    
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u5)
    model = keras.Model(inputs, outputs, name="UNet")
    model_optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        # binary_crossentropy
        loss="binary_crossentropy",
        optimizer = model_optimizer, 
        metrics = ['accuracy'])
    
    # model.summary()
    return model
##############################################################################
########################################
# 2D Standard
########################################

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):

    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_1', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = Dropout(dropout_rate, name='dp'+stage+'_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv'+stage+'_2', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(dropout_rate, name='dp'+stage+'_2')(x)

    return x

########################################

"""
Standard U-Net [Ronneberger et.al, 2015]
Total params: 7,759,521
"""
def U_Net(img_rows, img_cols, color_type=1, num_class=1):

    nb_filter = [32,64,128,256,512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_data_format() == 'channels_last':
      bn_axis = 3
      img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    else:
      bn_axis = 1
      img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    unet_output = Conv2D(num_class, (1, 1), activation='sigmoid', name='output', kernel_initializer = 'he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    model = Model(img_input, unet_output)

    return model

if __name__ == '__main__':
    
    UNetPlusPlus = UNetPlusPlus(512,512,1)
    UNetPlusPlus.summary()
    # plot_model(U_Net)

    # image_size = 512
    # input_1 = keras.Input(shape=(image_size, image_size, 1))
    # input_2 = keras.Input(shape=(image_size, image_size, 1))
    # input_3 = keras.Input(shape=(image_size, image_size, 1))
    # input_4 = keras.Input(shape=(image_size, image_size, 1))
    # input_5 = keras.Input(shape=(image_size, image_size, 1))
    
    # concatenate_inputs = layers.concatenate([input_1, input_2, input_3, input_4, input_5])
    
    # model = UNet(image_size = (512, 512, 5))
    # output = model(concatenate_inputs)
    
    # SL_UNet_model = keras.Model([input_1, input_2, input_3, input_4, input_5],
    #                             output,
    #                             name = 'structured_light_Unet')
    
    # model_optimizer = Adam(learning_rate=0.01)
    # SL_UNet_model.compile(
    #     # binary_crossentropy
    # #     loss = "mean_squared_error"
    #     loss="binary_crossentropy",
    #     optimizer = model_optimizer,
    #     metrics = ['accuracy'])
    
    # SL_UNet_model.summary()