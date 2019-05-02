import tensorflow as tf
from tensorflow.python import nn_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.utils import conv_utils
import numpy as np
from load_data import load_cifar, load_mnist, preprocess_cifar_data, preprocess_mnist_data


# Enable eager execution
tf.enable_eager_execution()


class RecurrentConvolutionalLayer(tf.keras.layers.Layer):

    PADDING_NAMES = ["SAME", "VALID"]

    PRECISION_NP= np.float32
    PRECISION_TF= tf.float32

    def __init__(self,
                 number_of_filters,
                 kernel_size,
                 execution_depth=3,
                 alpha=1e-3,
                 beta=0.75,
                 normalization_feature_maps=8,
                 strides=1,
                 initializer_forward=None,
                 initializer_recurrent=None,
                 initializer_bias=None,
                 padding="same",
                 **kwargs
                 ):

        self.built = False
        self.rank = 2
        self.number_of_filters = number_of_filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, "kernel_size")

        if not padding:
            raise ValueError("Padding should not be set to none")

        if padding.upper() not in self.PADDING_NAMES:
            raise ValueError("Padding must be set either to 'valid' or to 'same'")

        self.padding = padding.upper()
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.depth = execution_depth

        initializer = tf.contrib.layers.variance_scaling_initializer(dtype=RecurrentConvolutionalLayer.PRECISION_TF)
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

        if self.padding.upper() == 'VALID':
            self.unit_dim = (
                self.number_of_filters,
                (input_shape[1] - kernel_shape[0]) / float(self.strides[0]) + 1,
                (input_shape[2] - kernel_shape[1]) / float(self.strides[0]) + 1,
                1
            )
        elif self.padding.upper() == 'SAME':
            self.unit_dim = (
                self.number_of_filters,
                input_shape[1],
                input_shape[2],
                1
            )

        self.cell_states = np.zeros(
            self.unit_dim,
            dtype=RecurrentConvolutionalLayer.PRECISION_NP
        )

        recurrent_shape = tensor_shape.TensorShape([None, self.unit_dim[1], self.unit_dim[2], None])
        recurrent_kernel_shape = self.kernel_size + (1, 1)

        self.forward_kernel = self.add_weight(
            name="forward_kernel",
            shape=kernel_shape,
            initializer= self.initializer_forward,
            trainable=True,
            dtype=RecurrentConvolutionalLayer.PRECISION_TF
        )

        self.recurrent_kernel = self.add_weight(
            name="recurrent_kernel",
            shape=recurrent_kernel_shape,
            initializer=self.initializer_recurrent,
            trainable=True,
            dtype=RecurrentConvolutionalLayer.PRECISION_TF
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.number_of_filters,),
            initializer=self.initializer_bias,
            trainable=True,
            dtype=RecurrentConvolutionalLayer.PRECISION_TF
        )

        self.conv_forward = nn_ops.Convolution(
            input_shape=input_shape,
            filter_shape=self.forward_kernel.get_shape(),
            strides=self.strides,
            padding=self.padding
        )

        self.conv_recurrent= nn_ops.Convolution(
            recurrent_shape,
            filter_shape=self.recurrent_kernel.get_shape(),
            strides=self.strides,
            padding=self.padding
        )

        self.built = True

    def call(self, inputs, **kwargs):
        output_forward = self.conv_forward(inputs, self.forward_kernel)
        # TODO: How to invoke recurrent execution again?
        self.cell_states = np.zeros(self.cell_states.shape, dtype=RecurrentConvolutionalLayer.PRECISION_NP)

        outputs = None
        for _ in range(self.depth):
            output_recurrent = np.zeros(
                (1, self.unit_dim[1], self.unit_dim[2], self.number_of_filters),
                dtype=RecurrentConvolutionalLayer.PRECISION_NP
            )
            for num, feature_map in enumerate(output_recurrent.transpose([3, 2, 1, 0])):
                recurrent_result = self.conv_recurrent(np.asarray([feature_map]), self.recurrent_kernel)
                output_recurrent[:, :, :, num:num+1] = recurrent_result.numpy().transpose([3, 2, 1, 0])

            outputs = tf.add(output_forward, output_recurrent)
            outputs = nn_ops.bias_add(outputs, self.bias, data_format='NHWC')

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
    input_shape = (28, 28, 1)
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(RecurrentConvolutionalLayer(
        32,
        (3, 3),
        execution_depth=3,
        input_shape=input_shape
    ))
    model.add(RecurrentConvolutionalLayer(
        32,
        (3, 3),
        execution_depth=3,
        input_shape=input_shape
    ))

    model.add(tf.keras.layers.MaxPooling2D((2, 2)))

    model.add(RecurrentConvolutionalLayer(
        32,
        (3, 3),
        execution_depth=3,
        input_shape=input_shape
    ))
    model.add(RecurrentConvolutionalLayer(
        32,
        (3, 3),
        execution_depth=3,
        input_shape=input_shape
    ))

    model.add(tf.keras.layers.GlobalMaxPool2D())

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.summary()

    model.compile(
        optimizer=tf.train.AdagradOptimizer(learning_rate=.1),
        # loss='kullback_leibler_divergence',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    #
    # cifar_dict_1 = load_cifar('data_batch_1', dtype=RecurrentConvolutionalLayer.PRECISION_NP)
    # cifar_dict_2 = load_cifar('data_batch_2', dtype=RecurrentConvolutionalLayer.PRECISION_NP)
    # cifar_dict_3 = load_cifar('data_batch_3', dtype=RecurrentConvolutionalLayer.PRECISION_NP)
    # cifar_dict_4 = load_cifar('data_batch_4', dtype=RecurrentConvolutionalLayer.PRECISION_NP)
    #
    # training_data = np.concatenate((cifar_dict_1['data'], cifar_dict_2['data'], cifar_dict_3['data'], cifar_dict_4['data']))
    # training_data = preprocess_cifar_data(training_data, dtype=RecurrentConvolutionalLayer.PRECISION_NP)
    # training_labels = np.concatenate((cifar_dict_1['labels'], cifar_dict_2['labels'], cifar_dict_3['labels'], cifar_dict_4['labels']))


    mnist_dict = load_mnist('train', dtype=RecurrentConvolutionalLayer.PRECISION_NP)
    training_data = mnist_dict['data']
    training_data = preprocess_mnist_data(training_data, dtype=RecurrentConvolutionalLayer.PRECISION_NP)
    training_labels = mnist_dict['labels']

    model.fit(training_data, training_labels, epochs=5)

    # cifar_dict_test = load_cifar('test_batch', dtype=RecurrentConvolutionalLayer.PRECISION_NP)
    # test_data = preprocess_cifar_data(cifar_dict_test['data'], dtype=RecurrentConvolutionalLayer.PRECISION_NP)
    # test_labels = cifar_dict_test['labels']
    # model.evaluate(test_data, test_labels)

