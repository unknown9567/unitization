import collections
import numpy as np
from keras import (
    backend as K,
    layers as Kl,
    regularizers as Kr,
    models as Km
)
from .unitization import Unitization


AXIS = {
    'channels_first': 1, 'channels_last': -1
}[K.image_data_format()]


def get_convolutional_layer(
        filter, kernel_size, stride, name, weight_decay
    ):
    return Kl.Conv2D(
        filter,
        kernel_size,
        strides=stride,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=Kr.l2(weight_decay),
        name=name
    )


class Block(object):
    _block_count = 0
    def __init__(self, filter, stride, mode, normalization,
                 weight_decay, model_name=None):
        self.__dict__.update(locals())
        if mode not in ['33', '131']:
            raise Exception('Unknown mode: {}'.format(mode))
        self.normalization_class = {
            'batchnorm': Kl.BatchNormalization, 'unitization': Unitization
        }.get(normalization.lower(), None)
        if self.normalization_class is None:
            raise Exception('Unknown normalization: {}'.format(normalization))
        self.block_index = self.__class__._block_count
        self.__class__._block_count += 1

    def __call__(self, inputs, **kwargs):
        kwargs.setdefault('axis', AXIS)
        if self.model_name is None:
            prefix = 'block-{}-'.format(self.block_index)
        else:
            prefix = '{}-block-{}-'.format(self.model_name, self.block_index)

        layer_counts = collections.defaultdict(int)
        def get_name(cls):
            name = '{}-{}'.format(
                cls.__name__.lower()[:4], layer_counts[cls]
            )
            layer_counts[cls] += 1
            return prefix + name

        if self.mode == '33':
            filters = [self.filter] * 2
            kernel_sizes = [(3, 3), (3, 3)]
            strides = [self.stride] + [(1, 1)]
        else:
            filters = [self.filter] * 2 + [4 * self.filter]
            kernel_sizes = [(1, 1), (3, 3), (1, 1)]
            strides = [self.stride] + [(1, 1)] * 2

        if filters[-1] != K.int_shape(inputs)[kwargs['axis']] or self.stride != (1, 1):
            shortcut = get_convolutional_layer(
                filters[-1], (1, 1), self.stride,
                prefix+'shortcut', self.weight_decay
            )(inputs)
        else:
            shortcut = inputs

        tensor = inputs
        for filter, kernel_size, stride in zip(filters, kernel_sizes, strides):
            tensor = self.normalization_class(
                name=get_name(self.normalization_class), **kwargs
            )(tensor)

            tensor = Kl.Activation(
                'relu', name=get_name(Kl.Activation)
            )(tensor)

            tensor = get_convolutional_layer(
                filter, kernel_size, stride,
                get_name(Kl.Conv2D), self.weight_decay
            )(tensor)

            if 'image_size' in kwargs:
                if isinstance(stride, int):
                    kwargs['image_size'] //= stride**2
                else:
                    kwargs['image_size'] //= np.prod(stride)

        return Kl.Add(name=prefix+'add')([tensor, shortcut])


class ResNets(object):
    @staticmethod
    def _norm_relu(tensor, normalization, prefix, **kwargs):
        kwargs.setdefault('axis', AXIS)
        norm_cls = {
            'batchnorm': Kl.BatchNormalization,
            'unitization': Unitization
        }[normalization.lower()]
        tensor = norm_cls(
            name=prefix+norm_cls.__name__.lower()[:4], **kwargs
        )(tensor)
        return Kl.Activation('relu', name=prefix+'acti')(tensor)

    @staticmethod
    def _cifar_resnet(
            input_shape, num_classes, normalization, num_stacks,
            mode, name, weight_decay=5e-4, **kwargs
        ):
        if normalization.lower() not in ['batchnorm', 'unitization']:
            raise Exception('Unknown normalization: {}'.format(normalization))

        if normalization.lower() == 'batchnorm':
            kwargs.pop('image_size', None)
        else:
            kwargs.setdefault('image_size', 32**2)

        inputs = Kl.Input(input_shape, name=name+'-inputs')

        tensor = inputs
        tensor = get_convolutional_layer(
            16, (3, 3), (1, 1), name+'-pre-conv', weight_decay
        )(tensor)

        tensor = ResNets._norm_relu(
            tensor, normalization, name+'-pre-', **kwargs
        )

        filters = [16] * num_stacks \
            + [32] * num_stacks \
            + [64] * num_stacks
        strides = [(1, 1)] * num_stacks \
            + [(2, 2)] + [(1, 1)] * (num_stacks - 1) \
            + [(2, 2)] + [(1, 1)] * (num_stacks - 1)

        Block._block_count = 0
        for filter, stride in zip(filters, strides):
            tensor = Block(
                filter=filter,
                stride=stride,
                mode=mode,
                normalization=normalization,
                weight_decay=weight_decay,
                model_name=name
            )(tensor, **kwargs)
            if 'image_size' in kwargs:
                if isinstance(stride, int):
                    kwargs['image_size'] //= stride
                else:
                    kwargs['image_size'] //= np.prod(stride)

        tensor = ResNets._norm_relu(
            tensor, normalization, name+'-fin-', **kwargs
        )
        tensor = Kl.GlobalAvgPool2D(name=name+'-fin-pool')(tensor)
        outputs = Kl.Dense(
            num_classes,
            activation='softmax',
            kernel_initializer='he_normal',
            kernel_regularizer=Kr.l2(weight_decay),
            name=name+'-outputs'
        )(tensor)

        return Km.Model(inputs, outputs, name=name)

    @staticmethod
    def cifar_resnet_18(normalization, num_classes, **kwargs):
        return ResNets._cifar_resnet(
            input_shape=kwargs.pop('input_shape', (None, None, 3)),
            num_classes=num_classes,
            normalization=normalization,
            num_stacks=3,
            mode='33',
            name=kwargs.pop('name', 'cifar-resnet-18'),
            weight_decay=kwargs.pop('weight_decay', 5e-4),
            **kwargs
        )

    @staticmethod
    def cifar_resnet_110(normalization, num_classes, **kwargs):
        return ResNets._cifar_resnet(
            input_shape=kwargs.pop('input_shape', (None, None, 3)),
            num_classes=num_classes,
            normalization=normalization,
            num_stacks=18,
            mode='33',
            name=kwargs.pop('name', 'cifar-resnet-110'),
            weight_decay=kwargs.pop('weight_decay', 5e-4),
            **kwargs
        )

    @staticmethod
    def cifar_resnet_164(normalization, num_classes, **kwargs):
        return ResNets._cifar_resnet(
            input_shape=kwargs.pop('input_shape', (None, None, 3)),
            num_classes=num_classes,
            normalization=normalization,
            num_stacks=18,
            mode='131',
            name=kwargs.pop('name', 'cifar-resnet-164'),
            weight_decay=kwargs.pop('weight_decay', 5e-4),
            **kwargs
        )
