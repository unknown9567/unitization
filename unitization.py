import numpy as np
import tensorflow as tf
from keras import (
    backend as K,
    layers as Kl,
    initializers as Ki
)


class Unitization(Kl.BatchNormalization):
    def __init__(self, initial_alpha=0.0, image_size=None, scale_coe=1.0, **kwargs):
        super(Unitization, self).__init__(**kwargs)
        self.initial_alpha = initial_alpha
        self.image_size = image_size
        self.scale_coe = scale_coe

    def build(self, input_shape):
        # for the case of axis == -1
        self.axis = list(range(len(input_shape)))[self.axis]

        dim = input_shape[self.axis]
        self.alpha = self.add_weight(
            shape=(dim, ),
            name='alpha',
            initializer=Ki.Constant(self.initial_alpha),
            trainable=True
        )

        super(Unitization, self).build(input_shape)

    def call(self, inputs, training=None):
        axis = self.axis
        input_shape = K.int_shape(inputs)
        ndim = len(input_shape)
        dim = input_shape[axis]
        dtype = K.dtype(inputs)

        if ndim > 2:
            image_axes = list(ax for ax in range(1, ndim) if ax != axis)
            if self.image_size is not None:
                scale_squared_norm = tf.cast(1.0 / self.image_size, dtype=dtype)
            else:
                num_pixels = K.prod([K.shape(inputs)[ax] for ax in image_axes])
                scale_squared_norm = 1.0 / K.cast(num_pixels, dtype=dtype)
            if self.scale_coe != 1.0:
                scale_squared_norm /= self.scale_coe

        broadcast_shape = [1] * ndim
        broadcast_shape[axis] = dim

        def unitized_inference():
            broadcasted_moving_mean = K.reshape(
                self.moving_mean, broadcast_shape
            )
            broadcasted_moving_variance = K.reshape(
                self.moving_variance, broadcast_shape
            )

            broadcasted_moving_variance += self.epsilon
            scale = tf.rsqrt(broadcasted_moving_variance)
            centered_inputs = inputs - broadcasted_moving_mean
            if ndim > 2:
                squared_inputs = tf.reduce_mean(
                    centered_inputs**2, image_axes, True
                )
            else:
                squared_inputs = centered_inputs**2
            normalized_inputs = squared_inputs / broadcasted_moving_variance
            squared_norm = tf.reduce_sum(normalized_inputs, [axis], True)
            if ndim > 2:
                squared_norm *= scale_squared_norm

            alpha = K.reshape(self.alpha, broadcast_shape)
            scale *= alpha * tf.rsqrt(squared_norm + self.epsilon) + (1 - alpha)
            if self.scale:
                scale *= K.reshape(self.gamma, broadcast_shape)
            outputs = scale * centered_inputs

            if self.center:
                outputs += K.reshape(self.beta, broadcast_shape)

            return outputs

        if training in {0, False}:
            unitized_inputs = unitize_inference()

        else:
            reduction_axes = list(ax for ax in range(ndim) if ax != axis)
            mean = tf.reduce_mean(inputs, reduction_axes, False)
            broadcasted_mean = K.reshape(mean, broadcast_shape)
            centered_inputs = inputs - broadcasted_mean
            if ndim > 2:
                squared_inputs = tf.reduce_mean(
                    centered_inputs**2, image_axes, True
                )
            else:
                squared_inputs = centered_inputs**2
            broadcasted_variance = tf.reduce_mean(squared_inputs, [0], True)
            sample_size = K.prod([
                K.shape(inputs)[axis] for axis in reduction_axes
            ])
            sample_size = K.cast(sample_size, dtype=dtype)
            broadcasted_variance *= sample_size / (sample_size - (1.0 + self.epsilon))
            variance = tf.squeeze(broadcasted_variance, reduction_axes)

            self.add_update(
                [
                    K.moving_average_update(
                        self.moving_mean, mean, self.momentum
                    ),
                    K.moving_average_update(
                        self.moving_variance, variance, self.momentum
                    )
                ],
                inputs
            )

            broadcasted_variance += self.epsilon
            scale = tf.rsqrt(broadcasted_variance)

            normalized_inputs = squared_inputs / broadcasted_variance
            squared_norm = tf.reduce_sum(normalized_inputs, [axis], True)
            if ndim > 2:
                squared_norm *= scale_squared_norm
            alpha = K.reshape(self.alpha, broadcast_shape)
            scale *= alpha * tf.rsqrt(squared_norm + self.epsilon) + (1 - alpha)

            if self.scale:
                scale *= K.reshape(self.gamma, broadcast_shape)

            unitized_inputs = scale * centered_inputs

            if self.center:
                unitized_inputs += K.reshape(self.beta, broadcast_shape)

        return K.in_train_phase(
            unitized_inputs, unitized_inference, training=training
        )

    def get_config(self):
        config = {
            'initial_alpha': self.initial_alpha,
            'image_size': self.image_size,
            'scale_coe': self.scale_coe
        }
        base_config = super(Unitization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
