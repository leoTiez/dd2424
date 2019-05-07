import sys
import tensorflow as tf
import numpy as np
from load_data import load_mnist, preprocess_mnist_data

# Constants
# float64 is not allowed by all tf operations
PRECISION_TF = tf.float32
PRECISION_NP = np.float32
PADDING_LIST = ['SAME', 'VALID']


def rcl(
        input_data,
        num_input_channels,
        num_filter,
        filter_shape,
        num_of_data,
        depth=3,
        std=.03,
        alpha=1e-3,
        beta=.75,
        normalization_feature_maps=8,
        name='rcl'
):
    conv_filter_forward_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filter]
    conv_filter_recurrent_shape = [filter_shape[0], filter_shape[1], num_filter, num_filter]
    recurrent_cells_shape = [
        num_of_data,
        input_data.shape[1].value,
        input_data.shape[2].value,
        num_filter
    ]

    cell_states = tf.fill(dims=recurrent_cells_shape, value=0.0)
    output_init = tf.fill(dims=recurrent_cells_shape, value=0.0)

    forward_weights = tf.Variable(
        tf.truncated_normal(conv_filter_forward_shape, stddev=std, dtype=PRECISION_TF),
        trainable=True,
        name=name + '_forward'
    )

    recurrent_weights = tf.Variable(
        tf.truncated_normal(conv_filter_recurrent_shape, stddev=std, dtype=PRECISION_TF),
        trainable=True,
        name=name + '_recurrent'
    )

    bias = tf.Variable(
        tf.truncated_normal([num_filter], dtype=PRECISION_TF),
        trainable=True,
        name=name + '_bias'
    )

    forward_output = tf.nn.conv2d(input_data, forward_weights, [1, 1, 1, 1], padding='SAME')

    def loop_body(x, recurrent_states, output):
        recurrent_output = tf.nn.conv2d(recurrent_states, recurrent_weights, [1, 1, 1, 1], padding='SAME')

        output = tf.add(forward_output, recurrent_output)
        output += bias

        recurrent_states = tf.nn.local_response_normalization(
            tf.maximum(np.asarray(0.0).astype(PRECISION_NP), output),
            depth_radius=normalization_feature_maps,
            bias=1,
            alpha=alpha,
            beta=beta,
            name=name + '_lrn'
        )
        x += 1
        return x, recurrent_states, output

    results = tf.while_loop(
        lambda x, recurrent_states, output: x < depth,
        loop_body,
        [0, cell_states, output_init],
    )

    return results[2]


def convolutional_layer(
        input_data,
        filter_shape,
        num_input_channels,
        num_filter,
        std=.03,
        stride=(1, 1),
        padding='same',
        name='conv'
):
    conv_filter_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filter]
    strides = [1, stride[0], stride[1], 1]

    weights = tf.Variable(
        tf.truncated_normal(conv_filter_shape, stddev=std, dtype=PRECISION_TF),
        trainable=True,
        name=name + '_weights'
    )

    if padding.upper() not in PADDING_LIST:
        raise ValueError('Padding value is not understood')

    output = tf.nn.conv2d(input_data, weights, strides=strides, padding=padding.upper())

    return output


def pooling_layer(
        input_data,
        pool_shape,
        stride=(1, 1),
        padding='same'
):
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, stride[0], stride[1], 1]

    if padding.upper() not in PADDING_LIST:
        raise ValueError('Padding value is not understood')

    output = tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding=padding.upper())
    return output


def global_max_pooling_layer(
        input_data
):
    return tf.reduce_max(input_data, axis=[1, 2])


def softmax_layer(
        input_data,
        num_input_dim,
        num_output_dim,
        std=.03,
        name='softmax'
):
    weights = tf.Variable(
        tf.truncated_normal([num_input_dim, num_output_dim], stddev=std),
        trainable=True,
        name=name + '_weights'
    )

    bias = tf.Variable(
        tf.truncated_normal([num_output_dim], stddev=std),
        trainable=True,
        name=name + '_bias'
    )

    linear_trans = tf.matmul(input_data, weights) + bias
    return tf.nn.softmax(linear_trans), linear_trans


def accuracy(
        labels,
        result
):
    return tf.metrics.accuracy(labels=labels, predictions=result)[1]


class RCNN:
    def __init__(
            self,
            input_shape,
            output_shape=[None, 10],
            learning_rate=.1,
            num_filter=64,
            batch_size=100,
            shuffle_buffer_size=10000,
            conv_layer_filter_shape=[5, 5],
            rec_conv_layer_filter_shape=[3, 3],
            pooling_shape=[3, 3],
            pooling_stride_shape=[2, 2]
    ):
        self.input_placeholder = tf.placeholder(PRECISION_TF, input_shape)
        self.output_placeholder = tf.placeholder(PRECISION_TF, output_shape)

        # dropout probability placeholder
        self.rate_placeholder = tf.placeholder(PRECISION_TF)
        # number of data placeholder
        self.num_data_placeholder = tf.placeholder(tf.int64)

        # Create data set objects
        training_data_set = tf.data.Dataset.from_tensor_slices((
            self.input_placeholder,
            self.output_placeholder
        )).shuffle(buffer_size=shuffle_buffer_size).repeat().batch(batch_size=self.num_data_placeholder)
        test_data_set = tf.data.Dataset.from_tensor_slices((
            self.input_placeholder,
            self.output_placeholder,
        )).batch(batch_size=self.num_data_placeholder)

        # Create Iterator
        data_iterator = tf.data.Iterator.from_structure(training_data_set.output_types,
                                                         training_data_set.output_shapes)
        # Initialize iterators and get input and output placeholder variables
        self.train_init_op = data_iterator.make_initializer(training_data_set)
        self.test_init_op = data_iterator.make_initializer(test_data_set)
        # Defines the pipeline and creates a pointer to the next data point
        features, labels = data_iterator.get_next()

        # Net definition
        # First convolutional layer
        conv_layer = convolutional_layer(
            features,
            num_input_channels=input_shape[-1],
            filter_shape=conv_layer_filter_shape,
            num_filter=num_filter,
            name='conv_layer_1'
        )

        pooling_1 = pooling_layer(
            input_data=conv_layer,
            pool_shape=pooling_shape,
            stride=pooling_stride_shape
        )

        # Recurrent convolutional layers
        rcl_layer_1 = rcl(
            input_data=pooling_1,
            num_input_channels=num_filter,
            num_of_data=self.num_data_placeholder,
            filter_shape=rec_conv_layer_filter_shape,
            num_filter=num_filter,
            name='rcl_layer_1'
        )

        dropout_1 = tf.nn.dropout(
            rcl_layer_1,
            rate=self.rate_placeholder,
            name='drop_1'
        )

        rcl_layer_2 = rcl(
            input_data=dropout_1,
            num_of_data=self.num_data_placeholder,
            num_input_channels=num_filter,
            filter_shape=rec_conv_layer_filter_shape,
            num_filter=num_filter,
            name='rcl_layer_2'
        )

        pooling_2 = pooling_layer(
            input_data=rcl_layer_2,
            pool_shape=pooling_shape,
            stride=pooling_stride_shape
        )

        dropout_2 = tf.nn.dropout(
            pooling_2,
            rate=self.rate_placeholder,
            name='drop_2'
        )

        # Recurrent convolutional layer
        rcl_layer_3 = rcl(
            input_data=dropout_2,
            num_of_data=self.num_data_placeholder,
            num_input_channels=num_filter,
            filter_shape=rec_conv_layer_filter_shape,
            num_filter=num_filter,
            name='rcl_layer_3'
        )

        dropout_3 = tf.nn.dropout(
            rcl_layer_3,
            rate=self.rate_placeholder,
            name='drop_3'
        )

        rcl_layer_4 = rcl(
            input_data=dropout_3,
            num_of_data=self.num_data_placeholder,
            num_input_channels=num_filter,
            filter_shape=rec_conv_layer_filter_shape,
            num_filter=num_filter,
            name='rcl_layer_4'
        )

        # Max pooling layer
        global_max = global_max_pooling_layer(rcl_layer_4)

        # Flatten tensor for softmax layer
        flatten = tf.reshape(global_max, [-1, global_max.shape[-1].value])

        # Softmax layer
        result, linear_trans = softmax_layer(
            flatten,
            global_max.shape[-1].value,
            output_shape_[1]
        )

        # cross entropy and accuracy matrix
        self.cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=linear_trans, labels=labels)
        )

        tf.summary.scalar('loss', self.cross_entropy)

        self.optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cross_entropy)

        # Accuracy
        self.accuracy = accuracy(
            labels,
            result
        )

        tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()
        self.global_init_op = tf.global_variables_initializer()
        self.local_init_op = tf.local_variables_initializer()

    def train(
            self,
            training_data_features,
            training_data_labels,
            val_data_features,
            val_data_labels,
            batch_size=100,
            epochs=7,
            create_graph=True
    ):
        with tf.Session() as sess:
            if create_graph:
                writer = tf.summary.FileWriter('logs/.')
                writer.flush()
                writer.add_graph(sess.graph)

            train_writer = tf.summary.FileWriter('logs/train', sess.graph)
            test_writer = tf.summary.FileWriter('logs/test')
            # initialise the variables
            sess.run(self.global_init_op)
            sess.run(self.local_init_op)

            total_batch = int(training_data_features.shape[0] / batch_size)
            for epoch in range(epochs):
                avg_cost = 0
                sess.run(self.train_init_op, feed_dict={
                    self.input_placeholder: training_data_features,
                    self.output_placeholder: training_data_labels,
                    self.num_data_placeholder: batch_size
                })
                for i in range(total_batch):
                    progress = (i / float(total_batch - 1)) * 100
                    print '\r {:.1f}%'.format(progress), '\t{0}> '.format('#' * int(progress)),

                    accuracies, _, cost_ = sess.run([self.summaries, self.optimiser, self.cross_entropy],
                                                     feed_dict={
                                                         self.rate_placeholder: 0.2,
                                                         self.num_data_placeholder: batch_size,
                                                     })
                    avg_cost += cost_ / total_batch
                    train_writer.add_summary(accuracies)

                sess.run(self.test_init_op, feed_dict={
                    self.input_placeholder: val_data_features,
                    self.output_placeholder: val_data_labels,
                    self.num_data_placeholder: val_data_features.shape[0]

                })
                val_acc, accuracies = sess.run([self.accuracy, self.summaries],
                                                  feed_dict={
                                                      self.rate_placeholder: 0,
                                                      self.num_data_placeholder: val_data_features.shape[0]
                                                  })
                test_writer.add_summary(accuracies)
                print "\nEpoch:", (epoch + 1), \
                    "cost =", "{:.3f}".format(avg_cost), \
                    "test accuracy: {:.3f}".format(val_acc)


if __name__ == '__main__':
    # Setting the parameters
    input_shape_ = [None, 28, 28, 1]
    output_shape_ = [None, 10]
    learning_rate_ = .1
    epochs_ = 5
    # For me, setting the *_size to more than 2000 my system ran out of memory
    # For the large scale tests it should not be an issue anymore and the test size can be increased
    batch_size_ = 100
    test_data_size_ = 2000
    num_filter_ = 64
    buffer_size_ = 10000

    # Load and transform the data
    mnist_dict_ = load_mnist('train', dtype=PRECISION_NP)
    training_data_np_, training_labels_np_ = preprocess_mnist_data(
        mnist_dict_['data'],
        mnist_dict_['labels'],
        dtype=PRECISION_NP
    )
    training_data_ = training_data_np_[:1000]
    training_labels_ = training_labels_np_[:1000]
    test_data_ = training_data_np_[-test_data_size_:]
    test_labels_ = training_labels_np_[-test_data_size_:]

    rcnn = RCNN(input_shape_)

    rcnn.train(
        training_data_,
        training_labels_,
        test_data_,
        test_labels_
    )
