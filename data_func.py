import os
import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib.pyplot as plt


def parse_image_func(filename):
    
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.rgb_to_grayscale(image)
    image = image/255

    return image

def parser(filename_1, filename_2):
    
    image_1 = parse_image_func(filename_1)      
    image_2 = parse_image_func(filename_2)

    return (image_1, image_2)

def create_img_dataset(filepath):
    dataset = os.listdir(filepath)
    dataset = sorted(dataset)
    num= 0
    
    for img_name in dataset:
        data_path = os.path.join(filepath, img_name)
        dataset[num] = data_path
        num += 1
        
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    
    print(filepath, '"Create image dataset is Finish"')
    return dataset


def optimize_img_dataset(tra_dataset, gt_dataset,
                          BATCH_SIZE=4, TRA_RATIO=0.8):
    
    AUTOTUNE = tf.data.AUTOTUNE
    image_count = len(gt_dataset)
    BATCH_SIZE = BATCH_SIZE
    SHUFFLE_BUFFER_SIZE = image_count

    dataset = tf.data.Dataset.zip((tra_dataset, gt_dataset))
    ## 隨機弄亂
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True)
    dataset = dataset.map(parser)
    # dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    ################################################################
    DATASET_SIZE = len(dataset)
    TRA_RATIO = TRA_RATIO
    VAL_RATIO = (1-TRA_RATIO)/2
    TES_RATIO = (1-TRA_RATIO)/2
    tra_size = int(TRA_RATIO * DATASET_SIZE)
    val_size = int(VAL_RATIO * DATASET_SIZE)
    tes_size = int(TES_RATIO * DATASET_SIZE)
    
    train_dataset = dataset.take(tra_size)
    test_dataset = dataset.skip(tra_size)
    validation_dataset = test_dataset.take(val_size)
    test_dataset = test_dataset.take(tes_size)
    ################################################################
    # print(train_dataset.element_spec)
    print('train_dataset : %s' %len(train_dataset))
    print('validation_dataset : %s' %len(validation_dataset))
    print('test_dataset : %s' %len(test_dataset))

    
    return train_dataset, validation_dataset, test_dataset

def get_optimizer(STEPS_PER_EPOCH=3000):
    
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001,
        decay_steps=STEPS_PER_EPOCH*5,
        decay_rate=1,
        staircase=False)

    return tf.keras.optimizers.Adam(lr_schedule)


if __name__ == '__main__':
    
    tra_path = 'D:\\joy\\Structured_Light\\SL_data\\SL_dataset_version_5\\Image_Data'
    gt_path = 'D:\\joy\\Structured_Light\\SL_data\\SL_dataset_version_5\\Ground_Truth'
    
    x_train = create_img_dataset(tra_path)
    y_train = create_img_dataset(gt_path)
    
    ##################### testing ######################
    # dataset = tf.data.Dataset.zip((x_train, y_train))
    # dataset = dataset.shuffle(buffer_size=len(x_train))
    # dataset = dataset.take(10)
    # print(dataset)
    # dataset = dataset.map(parser)
    # # print(len(x_train))
    

    # for img_1, img_2 in dataset:
    #     # print(path_1, path_2)
        
    #     fig = plt.figure(figsize=(60,30))
    #     fig.subplots_adjust(hspace=0.2, wspace=0.2)
        
    #     ax = fig.add_subplot(1,3,1)
    #     ax.imshow(img_1, cmap='gray') 
    #     ax = fig.add_subplot(1,3,2)
    #     ax.imshow(img_2, cmap='gray')
    
    #####################################################

    train_dataset, validation_dataset, test_dataset = optimize_img_dataset(x_train, y_train,
                                                                            BATCH_SIZE=4,
                                                                            TRA_RATIO=0.9)
    

    


