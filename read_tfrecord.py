# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import numpy as np
import tensorflow_addons as tfa
import random
import time

# from VGG import VGG16, VGG16_V2
# from resnet20 import resnet_v1, lr_schedule

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# strategy = tf.distribute.MirroredStrategy(devices=['/gpu:2', '/gpu:3', '/gpu:4', '/gpu:5', '/gpu:6', '/gpu:7'])

eigvec = tf.constant([[-0.5675, 0.7192, 0.4009], [-0.5808, -0.0045, -0.8140], [-0.5836, -0.6948, 0.4203]], shape=[3, 3],
                     dtype=tf.float32)
eigval = tf.constant([55.46, 4.794, 1.148], shape=[3, 1], dtype=tf.float32)
mean_RGB = tf.constant([123.68, 116.779, 109.939], dtype=tf.float32)
std_RGB = tf.constant([58.393, 57.12, 57.375], dtype=tf.float32)

imageWidth = 224
imageHeight = 224
resize_min = 256
random_min_aspect = 0.75
random_max_aspect = 1 / 0.75
random_min_area = 0.08
random_angle = 7.
train_images = 100000

TRAIN_DIR = './tfrecord/train_tf/'
VALID_DIR = './tfrecord/valid_tf/'
TRAIN_FILE = [TRAIN_DIR + item for item in os.listdir(TRAIN_DIR)]
VALID_FILE = [VALID_DIR + item for item in os.listdir(VALID_DIR)]
random.shuffle(TRAIN_FILE)
random.shuffle(VALID_FILE)
num_classes = 100
batch_size = 128
epochs = 2
steps_per_epochs = train_images // batch_size
initial_learning_rate = 0.05
minimum_learning_rate = 0.0001
warm_iterations = steps_per_epochs
initial_lr = 0.02
initial_warmup_steps = 1000


def _parse_function(example_proto):
    feature = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature)
    image_decoded = tf.image.decode_jpeg(parsed_features['image'], channels=3)
    image_decoded = tf.cast(image_decoded, dtype=tf.float32)
    # Random crop the image
    shape = tf.shape(image_decoded)
    height, width = shape[0], shape[1]
    random_aspect = tf.random.uniform(shape=[], minval=random_min_aspect, maxval=random_max_aspect)
    random_area = tf.random.uniform(shape=[], minval=random_min_area, maxval=1.0)
    crop_width = tf.math.sqrt(
        tf.divide(
            tf.multiply(
                tf.cast(tf.multiply(height, width), tf.float32),
                random_area),
            random_aspect)
    )
    crop_height = tf.cast(crop_width * random_aspect, tf.int32)
    crop_height = tf.cond(crop_height < height, lambda: crop_height, lambda: height)
    crop_width = tf.cast(crop_width, tf.int32)
    crop_width = tf.cond(crop_width < width, lambda: crop_width, lambda: width)
    cropped = tf.image.random_crop(image_decoded, [crop_height, crop_width, 3])
    resized = tf.image.resize(cropped, [imageHeight, imageWidth])
    # Flip to add a little more random distortion in.
    flipped = tf.image.random_flip_left_right(resized)
    # Random rotate the image
    angle = tf.random.uniform(shape=[], minval=-random_angle, maxval=random_angle) * np.pi / 180
    rotated = tfa.image.rotate(flipped, angle)
    # Random distort the image
    distorted = tf.image.random_hue(rotated, max_delta=0.3)
    distorted = tf.image.random_saturation(distorted, lower=0.6, upper=1.4)
    distorted = tf.image.random_brightness(distorted, max_delta=0.3)
    # Add PCA noice
    alpha = tf.random.normal([3], mean=0.0, stddev=0.1)
    pca_noice = tf.reshape(tf.matmul(tf.multiply(eigvec, alpha), eigval), [3])
    distorted = tf.add(distorted, pca_noice)
    # Normalize RGB
    distorted = tf.subtract(distorted, mean_RGB)
    distorted = tf.divide(distorted, std_RGB)

    labels = tf.one_hot(parsed_features["label"], depth=num_classes)
    return distorted, labels


def train_input_fn():
    dataset_train = tf.data.TFRecordDataset(TRAIN_FILE)
    dataset_train = dataset_train.map(_parse_function, num_parallel_calls=4)
    dataset_train = dataset_train.shuffle(
        buffer_size=3200,
        reshuffle_each_iteration=True
    )
    # dataset_train = dataset_train.repeat(10)
    # repeat indefinitely
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.batch(batch_size)
    # dataset_train = dataset_train.prefetch(batch_size)
    return dataset_train


def _parse_test_function(example_proto):
    feature = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, feature)
    image_decoded = tf.image.decode_jpeg(parsed_features["image"], channels=3)
    image_decoded = tf.cast(image_decoded, dtype=tf.float32)
    shape = tf.shape(image_decoded)
    height, width = shape[0], shape[1]
    resized_height, resized_width = tf.cond(height < width,
                                            lambda: (resize_min, tf.cast(
                                                tf.multiply(tf.cast(width, tf.float64), tf.divide(resize_min, height)),
                                                tf.int32)),
                                            lambda: (tf.cast(
                                                tf.multiply(tf.cast(height, tf.float64), tf.divide(resize_min, width)),
                                                tf.int32), resize_min))
    image_resized = tf.image.resize(image_decoded, [resized_height, resized_width])
    # calculate how many to be center crop
    shape = tf.shape(image_resized)
    height, width = shape[0], shape[1]
    amount_to_be_cropped_h = (height - imageHeight)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - imageWidth)
    crop_left = amount_to_be_cropped_w // 2
    image_cropped = tf.slice(image_resized, [crop_top, crop_left, 0], [imageHeight, imageWidth, -1])
    # Normalize RGB
    image_valid = tf.subtract(image_cropped, mean_RGB)
    image_valid = tf.divide(image_valid, std_RGB)
    labels = tf.one_hot(parsed_features["label"], depth=num_classes)
    return image_valid, labels


def val_input_fn():
    dataset_valid = tf.data.TFRecordDataset(VALID_FILE)
    dataset_valid = dataset_valid.map(_parse_test_function, num_parallel_calls=4)
    dataset_valid = dataset_valid.shuffle(
        buffer_size=3200,
        reshuffle_each_iteration=True
    )
    dataset_valid = dataset_valid.batch(batch_size)
    dataset_valid = dataset_valid.prefetch(batch_size)
    return dataset_valid


if __name__ == '__main__':
    dataset_train = train_input_fn()
    # # test code
    # for images, labels in dataset_train:
    #     for image, label in zip(images, labels):
    #         image_numpy = image.numpy()
    #         image = tf.keras.preprocessing.image.array_to_img(image_numpy)
    #         label = tf.argmax(label).numpy()
    #         print('label:', label)
    #         image.show()
    dataset_val = val_input_fn()

    # model = VGG16((imageWidth, imageHeight, 3), num_classes)
    # model.compile(optimizer=tf.optimizers.SGD(learning_rate=0.005, momentum=0.9),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # model.fit(
    #     dataset_train,
    #     validation_data=dataset_val,
    #     epochs=epochs,
    #     steps_per_epoch=steps_per_epochs,
    #     callbacks=tensorboard_cbk
    # )
    # for images, labels in dataset_val:
    #     image_slice = tf.gather(images, tmp)
    #     image_list_numpy = image_slice.numpy()
    #     image_list = list(image_list_numpy)
    #     final = np.array(image_list)
    #     np.savez('./test.npz', final=final)
    #     for image, label in zip(images, labels):
    #         label = tf.argmax(label).numpy()
    #         print(label)
