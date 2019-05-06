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
        normalization_feature_maps=4,
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


if __name__ == '__main__':
    # Setting the parameters
    input_shape_ = [None, 28, 28, 1]
    output_shape_ = [None, 10]
    learning_rate_ = .1
    epochs_ = 5
    # For me, setting the *_size to more than 2000 my system ran out of memory
    # For the large scale tests it should not be an issue anymore and the test size can be increased
    batch_size_ = 12
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
    training_data_ = training_data_np_[:5000]
    training_labels_ = training_labels_np_[:5000]
    test_data_ = training_data_np_[-test_data_size_:]
    test_labels_ = training_labels_np_[-test_data_size_:]

    # Create input and output placeholder
    input_placeholder_ = tf.placeholder(PRECISION_TF, input_shape_)
    output_placeholder_ = tf.placeholder(PRECISION_TF, output_shape_)

    # Create data set objects
    training_data_set_ = tf.data.Dataset.from_tensor_slices((
        input_placeholder_,
        output_placeholder_
    )).shuffle(buffer_size=buffer_size_).repeat().batch(batch_size=batch_size_)
    test_data_set_ = tf.data.Dataset.from_tensor_slices((
        input_placeholder_,
        output_placeholder_
    )).batch(batch_size_)

    # Create Iterator
    data_iterator_ = tf.data.Iterator.from_structure(training_data_set_.output_types, training_data_set_.output_shapes)
    # Initialize iterators and get input and output placeholder variables
    train_init_op_ = data_iterator_.make_initializer(training_data_set_)
    test_init_op_ = data_iterator_.make_initializer(test_data_set_)
    # Defines the pipeline and creates a pointer to the next data point
    features_, labels_ = data_iterator_.get_next()

    # dropout probability placeholder
    rate_placeholder_ = tf.placeholder(PRECISION_TF)
    # number of data placeholder
    num_data_placeholder_ = tf.placeholder(PRECISION_TF)

    # Net definition
    # First convolutional layer
    conv_layer_ = convolutional_layer(
        features_,
        num_input_channels=1,
        filter_shape=(5, 5),
        num_filter=num_filter_,
        name='conv_layer_1'
    )

    # First pooling layer
    # Size 3 and stride 2, as proposed in the paper
    pooling_shape_1_ = (3, 3)
    striding_shape_1_ = (2, 2)

    pooling_1_ = pooling_layer(
        input_data=conv_layer_,
        pool_shape=pooling_shape_1_,
        stride=striding_shape_1_
    )

    # Recurrent convolutional layers
    rcl_layer_1_ = rcl(
        input_data=pooling_1_,
        num_input_channels=num_filter_,
        num_of_data=num_data_placeholder_,
        filter_shape=(3,3),
        num_filter=num_filter_,
        name='rcl_layer_1'
    )

    dropout_1_ = tf.nn.dropout(
        rcl_layer_1_,
        rate=rate_placeholder_,
        name='drop_1'
    )

    rcl_layer_2_ = rcl(
        input_data=dropout_1_,
        num_of_data=num_data_placeholder_,
        num_input_channels=num_filter_,
        filter_shape=(3, 3),
        num_filter=num_filter_,
        name='rcl_layer_2'
    )

    # Second pooling layer
    # Size 3 and stride 2, as proposed in the paper
    pooling_shape_2_ = (3, 3)
    striding_shape_2_ = (2, 2)


    pooling_2_ = pooling_layer(
        input_data=rcl_layer_2_,
        pool_shape=pooling_shape_2_,
        stride=striding_shape_2_
    )

    dropout_2_ = tf.nn.dropout(
        pooling_2_,
        rate=rate_placeholder_,
        name='drop_2'
    )

    # Recurrent convolutional layer
    rcl_layer_3_ = rcl(
        input_data=pooling_2_,
        num_of_data=num_data_placeholder_,
        num_input_channels=num_filter_,
        filter_shape=(3, 3),
        num_filter=num_filter_,
        name='rcl_layer_3'
    )

    dropout_3_ = tf.nn.dropout(
        rcl_layer_3_,
        rate=rate_placeholder_,
        name='drop_3'
    )

    rcl_layer_4_ = rcl(
        input_data=dropout_3_,
        num_of_data=num_data_placeholder_,
        num_input_channels=num_filter_,
        filter_shape=(3, 3),
        num_filter=num_filter_,
        name='rcl_layer_4'
    )

    # Max pooling layer
    global_max_ = global_max_pooling_layer(rcl_layer_4_)

    # Flatten tensor for softmax layer
    flatten_ = tf.reshape(global_max_, [-1, global_max_.shape[-1].value])

    # Softmax layer
    result_, linear_trans_ = softmax_layer(
        flatten_,
        global_max_.shape[-1].value,
        output_shape_[1]
    )

    # cross entropy and accuracy matrix
    cross_entropy_ = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=linear_trans_, labels=labels_)
    )

    tf.summary.scalar('loss', cross_entropy_)

    optimiser_ = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cross_entropy_)

    # Accuracy
    accuracy_ = accuracy(
        labels_,
        result_
    )

    tf.summary.scalar('accuracy', accuracy_)

    summaries = tf.summary.merge_all()
    global_init_op_ = tf.global_variables_initializer()
    local_init_op_ = tf.local_variables_initializer()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess_:
        #
        # writer = tf.summary.FileWriter('logs/.')
        # writer.add_graph(sess_.graph)
        # writer.flush()

        train_writer = tf.summary.FileWriter('logs/train', sess_.graph)
        test_writer = tf.summary.FileWriter('logs/test')
        # initialise the variables
        sess_.run(global_init_op_)
        sess_.run(local_init_op_)

        total_batch_ = int(training_data_.shape[0] / batch_size_)
        for epoch_ in range(epochs_):
            avg_cost_ = 0
            sess_.run(train_init_op_, feed_dict={
                input_placeholder_: training_data_,
                output_placeholder_: training_labels_
            })
            for i in range(total_batch_):
                progress = (i / float(total_batch_-1)) * 100
                print '\r {:.1f}%'.format(progress), '\t{0}> '.format('#' * int(progress)),

                accuracies, _, cost_ = sess_.run([summaries, optimiser_, cross_entropy_],
                                     feed_dict={
                                         rate_placeholder_: 0.2,
                                         num_data_placeholder_: batch_size_
                                     })
                avg_cost_ += cost_ / total_batch_
                train_writer.add_summary(accuracies)

            sess_.run(test_init_op_, feed_dict={
                input_placeholder_: test_data_,
                output_placeholder_: test_labels_
            })
            test_acc_ = sess_.run(accuracy_,
                                  feed_dict={
                                      rate_placeholder_: 0,
                                      num_data_placeholder_: batch_size_
                                  })
            #test_writer.add_summary(accuracies)
            print "\nEpoch:", (epoch_ + 1), \
                "cost =", "{:.3f}".format(avg_cost_),\
                "test accuracy: {:.3f}".format(test_acc_)

        sess_.run(test_init_op_, feed_dict={
            input_placeholder_: test_data_,
            output_placeholder_: test_labels_,
        })
        print "\nTraining complete!"
        print sess_.run(accuracy_,
                        feed_dict={
                            rate_placeholder_: 0,
                            num_data_placeholder_: batch_size_
                        })
        test_writer.close()
        train_writer.close()
