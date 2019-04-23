import tensorflow as tf
from tensorflow.python import nn_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils

import numpy as np

# Enable eager execution
tf.enable_eager_execution()


class RecurrentConvolutionalLayer(tf.keras.layers.Layer):

    PADDING_NAMES = ["SAME", "VALID"]

    def __init__(self,
                 number_of_filters,
                 kernel_size,
                 alpha=1e-3,
                 beta=0.75,
                 normalization_feature_maps=8,
                 strides=1,
                 initializer_forward=None,
                 initializer_recurrent=None,
                 initializer_bias=None,
                 padding="valid",
                 **kwargs
                 ):

        self.rank = 2
        self.number_of_filters = number_of_filters
        # self.number_of_units = int(number_of_units)
        # self.cell_states = np.zeros((number_of_units, 1))
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, "kernel_size")

        if not padding:
            raise ValueError("Padding should not be set to none")

        if padding.upper() not in self.PADDING_NAMES:
            raise ValueError("Padding must be set either to 'valid' or to 'valid'")

        self.padding = padding.upper()
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')

        initializer = tf.contrib.layers.variance_scaling_initializer(dtype=tf.float32)
        if initializer_forward:
            self.initializer_forward = tf.keras.initializers.get(initializer_forward)
        else:
            self.initializer_forward = initializer

        if initializer_recurrent:
            self.initializer_recurrent = tf.keras.initializers.get(initializer_recurrent)
        else:
            self.initializer_recurrent = initializer

        if initializer_bias:
            self.initializer_bias = tf.keras.initializers.get(initializer_bias)
        else:
            self.initializer_bias = initializer

        # hyperparameters for local response normalization
        self.alpha = alpha
        self.beta = beta
        if normalization_feature_maps > self.number_of_filters:
            raise ValueError("Number of normalization feature maps must be smaller that the number of filters")
        self.normalization_feature_maps = normalization_feature_maps

        super(RecurrentConvolutionalLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        # assuming that channels are always at last
        input_dim = int(input_shape[-1])

        kernel_shape = self.kernel_size + (input_dim, self.number_of_filters)

        self.number_of_units = ((input_shape[1] - kernel_shape[0]) / float(self.strides[0]) + 1)*\
                               ((input_shape[2] - kernel_shape[1]) / float(self.strides[0]) + 1)

        self.cell_states = np.zeros((self.number_of_units, self.number_of_filters))

        self.forward_kernel = self.add_weight(
            name="forward_kernel",
            shape=kernel_shape,
            initializer= self.initializer_forward,
            trainable=True
        )

        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=kernel_shape,
            initializer=self.initializer_recurrent,
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.number_of_filters, 1),
            initializer=self.initializer_bias,
            trainable=True
        )

        self.conv_forward = nn_ops.Convolution(
            input_shape=input_shape,
            filter_shape=self.forward_kernel.get_shape(),
            strides=self.strides,
            padding=self.padding
        )

        self.conv_recurrent= nn_ops.Convolution(
            input_shape,
            filter_shape=self.recurrent_kernel.get_shape(),
            strides=self.strides,
            padding=self.padding
        )

        self.built = True

    def call(self, inputs, **kwargs):
        output_forward = self.conv_forward(inputs, self.conv_forward)
        # TODO: How to invole recurrent execution again?
        output_recurrent = np.dot(self.cell_states, self.conv_recurrent)

        outputs = output_forward + output_recurrent
        outputs = nn_ops.bias_add(outputs, self.bias, data_format='NCHW')

        # apply local response normalization and relu as proposed in the paper
        self.cell_states = tf.nn.local_response_normalization(
            tf.maximum(0, outputs),
            depth_radius=self.normalization_feature_maps,
            bias=1,
            alpha=self.alpha,
            beta=self.beta,
            name="lrn"
        )

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = []

        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding.lower(),
                stride=self.strides[i]
            )
            new_space.append(new_dim)
        return tensor_shape.TensorShape(
            [input_shape[0]] +
            new_space +
            [self.number_of_filters]
        )

if __name__ == "__main__":
    model = tf.keras.models.Sequential()
    model.add(RecurrentConvolutionalLayer(32, (3, 3), input_shape=(28, 28, 1)))
    model.summary()


