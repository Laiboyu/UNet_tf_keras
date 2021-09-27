import os
import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

from keras.models import load_model
from keras import models, optimizers
from keras.optimizers import Adam
from PIL import Image

from loss_func import *
from plot_func import *
from data_func import *
from segmentation_models import UNet, UNetPlusPlus



print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
# print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
#%%
######################## System path #############################

data_folder_path = "Folder path to store Images"
model_folder_path = "Folder path to store Model and Weights"
log_dir_1 = "Folder path to store parameter of Tensorboard"

img_path = os.path.join(data_folder_path, 'Image_Data')
gt_path = os.path.join(data_folder_path, 'Ground_Truth')

weights_path = os.path.join(model_folder_path, 'best_weights.hdf5')
model_path = os.path.join(model_folder_path, 'UNet.hdf5')

#################################################################
x_train = create_img_dataset(img_path)
y_train = create_img_dataset(gt_path)

BATCH_SIZE = 2
TRA_RATIO = 0.9

train_dataset, validation_dataset, test_dataset = optimize_img_dataset(x_train, y_train, BATCH_SIZE=BATCH_SIZE, TRA_RATIO=TRA_RATIO)

steps_per_epoch = len(train_dataset)
model_optimizer = get_optimizer(STEPS_PER_EPOCH = steps_per_epoch)


print(train_dataset.element_spec)
print(len(train_dataset))
print(len(test_dataset))
print(len(validation_dataset))

"""
Select the architecture of model
"""
model = UNet(512, 512, 1)

steps_per_epoch = len(train_dataset)
model_optimizer = get_optimizer(STEPS_PER_EPOCH = steps_per_epoch)
# model_optimizer = Adam(learning_rate=0.001)
model.compile(
    
    loss = bce_dice_loss,
    optimizer = model_optimizer,
    metrics = ['accuracy', 'binary_crossentropy', dice_coef])


model.summary()

#%%
# Load the TensorBoard notebook extension
tensorboard_callback_1 = tf.keras.callbacks.TensorBoard(log_dir=log_dir_1, histogram_freq=1)

# ModelCheckpoint callback - save best weights
checkpoint_1 = tf.keras.callbacks.ModelCheckpoint(filepath=weights_path,
                                                  save_best_only=True,
                                                  verbose=1)

# EarlyStopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=10,
                                              restore_best_weights=True,
                                              mode='min')


EPOCHS = 30
history = model.fit(train_dataset, 
                    epochs=EPOCHS, 
                    validation_data=(validation_dataset),
                    callbacks=[tensorboard_callback_1, checkpoint_1, early_stop])

# 保存權重 - Save the Weights and Model
model.save(model_path) 

#%%

model = UNet(512, 512, 1)

steps_per_epoch = int(2664)
model_optimizer = get_optimizer(STEPS_PER_EPOCH = steps_per_epoch)
# model_optimizer = Adam(learning_rate=0.001)
model.compile(
    
    loss = bce_dice_loss,
    optimizer = model_optimizer,
    metrics = ['accuracy', 'binary_crossentropy', dice_coef])


model_path = "D:\\joy\\Structured_Light\\SL_model\\UNet\\one_to_one\\version_2\\best_weights.hdf5"
model.load_weights(model_path)
model.summary()
# result = model.predict(test_dataset)


#%%%
num = 0
BATCH_SIZE=2
for images, label in train_dataset:

    for batch_num in np.arange(0, BATCH_SIZE):
        
        result = model.predict(images)
        
        fig = plt.figure(figsize=(60,30))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        
        ax = fig.add_subplot(1,3,1)
        ax.imshow(images[batch_num], cmap='gray') 
        ax = fig.add_subplot(1,3,2)
        ax.imshow(result[batch_num], cmap='gray')
        ax = fig.add_subplot(1,3,3)
        ax.imshow(label[batch_num], cmap='gray')
        
        img_name = str(num+1).rjust(4,'0')+'.png'
        image_path = os.path.join("D:\\joy\\Structured_Light\\SL_data\\Prediction\\test\\2_2\\1", img_name)
        result_path = os.path.join("D:\\joy\\Structured_Light\\SL_data\\Prediction\\test\\2_2\\2", img_name)
        label_path = os.path.join("D:\\joy\\Structured_Light\\SL_data\\Prediction\\test\\2_2\\3", img_name)
        
        cv.imwrite(image_path, np.array(images[batch_num])*255.0)
        cv.imwrite(result_path, np.array(result[batch_num])*255.0)
        cv.imwrite(label_path, np.array(label[batch_num])*255.0)
        # fig.savefig(image_path)
        
        print("one of the comparison is compelete")
        num += 1
        

        
        





