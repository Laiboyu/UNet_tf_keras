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


def UNet(img_rows, img_cols, color_type=1, num_class=1):
    f = [16, 32, 64, 128, 256, 512]
    inputs = Input(shape=(img_rows, img_cols, color_type))
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
    model = Model(inputs, outputs, name="UNet")
    model_optimizer = Adam(learning_rate=0.001)
    
    model.compile(
        # binary_crossentropy
        loss="binary_crossentropy",
        optimizer = model_optimizer, 
        metrics = ['accuracy'])
    
    # model.summary()
    return model

if __name__ == '__main__':
    
    model = UNet(512,512,1)
    model.summary()
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