import numpy as np
import tensorflow as tf
from imblearn.over_sampling import RandomOverSampler
from imblearn.tensorflow import balanced_batch_generator
from keras.utils.data_utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class BalancedDataGenerator(Sequence):

    def __init__(self, x, y, datagen, classes_num=2, batch_size=32):
        self.datagen = datagen
        self.batch_size = batch_size
        self.classes_num = classes_num
        datagen.fit(x)
        s = x.shape
        reshaped_x = x.reshape((x.shape[0], -2))
        self.gen, self.steps_per_epoch = balanced_batch_generator(reshaped_x, y,
                                                                  sampler=RandomOverSampler(),
                                                                  batch_size=batch_size,
                                                                  keep_sparse=True)
        self._shape = (self.steps_per_epoch * batch_size, *x.shape[1:])


    def __len__(self):
        return self.steps_per_epoch


    def __getitem__(self, idx: int):
        """
        Returns next batch of images
        :param idx:
        :return: an instance of the NumpyArrayIterator class
        """
        x_batch, y_batch = self.gen.__next__()
        y_batch = tf.keras.utils.to_categorical(y_batch, num_classes=self.classes_num)
        x_batch = x_batch.reshape(-1, *self._shape[1:])
        return self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()