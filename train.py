import os
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD


def reset(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def train_on_cifar(dataset, model, seed, batch_size=128, epochs=200):
    if dataset not in ['CIFAR-10', 'CIFAR-100']:
        raise Exception('Unknown dataset: {}'.format(dataset))
    dataset = {'CIFAR-10': cifar10, 'CIFAR-100': cifar100}[dataset]
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    axis = {'channels_first': 1, 'channels_last': -1}[K.image_data_format()]
    if x_train.shape[axis] != 3:
        permutation = list(len(x_train.shape))
        permutation[1], permutation[-1] = permutation[-1], permutation[1]
        x_train = np.transpose(x_train, permutation)
        x_test = np.transpose(x_test, permutation)
    mean = np.mean(x_train, axis=0, keepdims=True)
    var = np.var(x_train, axis=0, keepdims=True)
    x_train = ((x_train - mean) / var).astype(np.float32)
    x_test = ((x_test - mean) / var).astype(np.float32)
    num_classes = np.max(y_train) + 1
    y_train = np.array([
        [1 if j == y_train[i] else 0 for j in range(num_classes)]
        for i in range(len(y_train))
    ]).astype(np.float32)
    y_test = np.array([
        [1 if j == y_test[i] else 0 for j in range(num_classes)]
        for i in range(len(y_test))
    ]).astype(np.float32)

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=.1, momentum=.9, nesterov=True),
                  metrics=['acc'])

    def scheduler(epoch):
        if epoch <= 60:
            return 0.05
        if epoch <= 120:
            return 0.01
        if epoch <= 160:
            return 0.002
        return 0.0004
    callbacks = [LearningRateScheduler(scheduler)]

    reset(seed)

    generator = ImageDataGenerator(
        horizontal_flip=True,
        width_shift_range=0.125,
        height_shift_range=0.125,
        fill_mode='constant', cval=0.
    )

    generator.fit(x_train)
    steps_per_epoch = (len(x_train) + batch_size - 1) // batch_size
    return model.fit_generator(
        generator.flow(x_train, y_train, batch_size, seed=seed),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=(x_test, y_test)
    )
