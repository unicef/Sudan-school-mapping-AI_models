import logging
from typing import List
import tensorflow as tf


"""
An implementation of the learning rate scheduler found at:
https://www.linkedin.com/pulse/some-tricks-handling-imbalanced-dataset-image-m-farhan-tandia/
"""
class LossLearningRateScheduler(tf.keras.callbacks.History):
    def __init__(self, base_lr: float, lookback_epochs: int, spike_epochs: List=None, spike_multiple: int=10,
                 decay_threshold: float=0.002, decay_multiple: float=0.7, loss_type: str='val_loss'):
        """
        :param base_lr: the starting learning rate
        :param lookback_epochs: the number of epochs in the past to compare with the loss function at the current epoch 
        to determine if progress is being made.
        :param spike_epochs: a list of epochs, for each epoch in the list the learning rate should be multiplied by spike_multiple value
        :param spike_multiple: how many times should the learning rate be multipled for epochs specified in the spike_epochs list
        :param decay_threshold: increase or decreases the lookback_epochs value.
        :param decay_multiple: if the loss function has not been improved by a factor of decay_threshold * lookback_epochs, this value 
        will be applied to the learning rate. 
        :param loss_type: what type of loss to monitor
        """
        super(LossLearningRateScheduler, self).__init__()
        self.base_lr = base_lr
        self.lookback_epochs = lookback_epochs
        self.spike_epochs = spike_epochs
        self.spike_multiple = spike_multiple
        self.decay_threshold = decay_threshold
        self.decay_multiple = decay_multiple
        self.loss_type = loss_type


    def on_epoch_begin(self, epoch, logs=None):
        if len(self.epoch) > self.lookback_epochs:
            current_lr = tf.keras.backend.get_value(self.model.optimizer.lr)
            target_loss = self.history[self.loss_type]
            loss_diff =  target_loss[-int(self.lookback_epochs)] - target_loss[-1]
            if loss_diff <= np.abs(target_loss[-1]) * (self.decay_threshold * self.lookback_epochs):
                logging.debug(' '.join(('Changing learning rate from', str(current_lr), 'to', str(current_lr * self.decay_multiple))))
                tf.keras.backend.set_value(self.model.optimizer.lr, current_lr * self.decay_multiple)
                current_lr = current_lr * self.decay_multiple
            else:
                logging.debug(' '.join(('Learning rate:', str(current_lr))))
            if self.spike_epochs is not None and len(self.epoch) in self.spike_epochs:
                logging.debug(' '.join(('Learning rate spike, from', str(current_lr), 'to', str(current_lr * self.spike_multiple))))
                tf.keras.backend.set_value(self.model.optimizer.lr, current_lr * self.spike_multiple)
        else:
            logging.debug(' '.join(('Setting learning rate to', str(self.base_lr))))
            tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)

        return tf.keras.backend.get_value(self.model.optimizer.lr)