############# Prediction ##############

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
from data_func_2 import *
from segmentation_models import UNet, UNetPlusPlus



print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
# print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

######################## System path #############################

system_path = "D:\\joy\\Structured_Light\\SL_code\\comparison_of_depth_construction\\5\\right"

im_path = os.path.join(system_path, 'img')
gt_path = os.path.join(system_path, 'gt')

#################################################################
x_train = create_img_dataset(im_path)
y_train = create_img_dataset(gt_path)

BATCH_SIZE = 2
TRA_RATIO = 1

train_dataset, validation_dataset, test_dataset = optimize_img_dataset(x_train, y_train, BATCH_SIZE=BATCH_SIZE, TRA_RATIO=TRA_RATIO)

steps_per_epoch = len(train_dataset)
model_optimizer = get_optimizer(STEPS_PER_EPOCH = steps_per_epoch)

# print(train_dataset)
print(train_dataset.element_spec)
print(len(train_dataset))
print(len(test_dataset))
print(len(validation_dataset))
##################################################################

#%%

model = UNet(512, 512, 1)

steps_per_epoch = int(2664)
model_optimizer = get_optimizer(STEPS_PER_EPOCH = steps_per_epoch)
# model_optimizer = Adam(learning_rate=0.001)
model.compile(
    
    loss = bce_dice_loss,
    optimizer = model_optimizer,
    metrics = ['accuracy', 'binary_crossentropy', dice_coef])


model_path = "D:\\joy\\Structured_Light\\SL_model\\UNet\\one_to_one\\version_3\\best_weights.hdf5"
model.load_weights(model_path)
model.summary()
# result = model.predict(test_dataset)



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
        # image_path = os.path.join("D:\\joy\\Structured_Light\\SL_data\\Prediction\\test\\2_2\\1", img_name)
        result_path = os.path.join(system_path, 'prediction', img_name)
        label_path = os.path.join(system_path, 'comparison', img_name)
        
        # cv.imwrite(image_path, np.array(images[batch_num])*255.0)
        cv.imwrite(result_path, np.array(result[batch_num])*255.0)
        # cv.imwrite(label_path, np.array(label[batch_num])*255.0)
        fig.savefig(label_path)
        
        print("one of the comparison is compelete")
        num += 1
        
#%%

image_path = os.path.join('D:\\joy\\Structured_Light\\SL_data\\predict\\test\\image', '1.png')
label_path = os.path.join('D:\\joy\\Structured_Light\\SL_data\\predict\\test\\label', '1.png')
result_path = os.path.join('D:\\joy\\Structured_Light\\SL_data\\predict\\test\\result', '1.png')


# tf.keras.utils.save_img(image_path, images[0])
img = np.array(images[0])*255.0
# plt.imshow(img)
cv.imwrite(image_path, np.array(images[0])*255.0)
cv.imwrite(label_path, np.array(label[0])*255.0)
cv.imwrite(result_path, np.array(label[0])*255.0)

print(img.shape)

        
        





