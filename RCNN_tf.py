import tensorflow as tf
import numpy as np
from load_data import load_mnist, preprocess_mnist_data

# float64 is not allowed by all tf operations
PRECISION_TF = tf.float32
PRECISION_NP = np.float32


def rcl(
        input_data,
        num_input_channels,
        num_filter,
        filter_shape,
        depth=3,
        std=.03,
        alpha=1e-3,
        beta=.75,
        normalization_feature_maps=8,
        name='rcl'
):
    conv_filter_forward_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filter]
    conv_filter_recurrent_shape = [filter_shape[0], filter_shape[1], num_filter, num_filter]
    recurrent_cells_shape = [1, input_data.shape[1], input_data.shape[2], num_filter]

    cell_states = tf.Variable(
        tf.zeros(recurrent_cells_shape, dtype=PRECISION_TF),
        trainable=False,
        name=name + '_cells'
    )

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

    output = None
    for _ in range(depth):
        recurrent_output = tf.nn.conv2d(cell_states, recurrent_weights, [1, 1, 1, 1], padding='SAME')

        output = tf.add(forward_output, recurrent_output)
        output += bias

        cell_states = tf.nn.local_response_normalization(
            tf.maximum(np.asarray(0.0).astype(PRECISION_NP), output),
            depth_radius=normalization_feature_maps,
            bias=1,
            alpha=alpha,
            beta=beta,
            name=name + '_lrn'
        )

    return output


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


if __name__ == '__main__':
    input_shape_ = [None, 28, 28, 1]
    output_shape_ = [None, 10]
    learning_rate_ = .1
    epochs_ = 1
    batch_size_ = 100
    test_data_size_ = 5000

    mnist_dict_ = load_mnist('train', dtype=PRECISION_NP)
    training_data_ = mnist_dict_['data']
    training_data_, training_labels_ = preprocess_mnist_data(
        training_data_,
        mnist_dict_['labels'],
        dtype=PRECISION_NP
    )
    training_data_ = training_data_[:-test_data_size_]
    training_labels_ = training_labels_[:-test_data_size_]
    test_data_ = training_data_[-test_data_size_:]
    test_labels_ = training_labels_[-test_data_size_:]

    input_placeholder_ = tf.placeholder(PRECISION_TF, input_shape_)
    output_placeholder_ = tf.placeholder(PRECISION_TF, output_shape_)

    rcl_layer_1_ = rcl(
        input_data=input_placeholder_,
        num_input_channels=1,
        filter_shape=(3,3),
        num_filter=32,
        name='rcl_layer_1'
    )

    flattern_dim_ = input_shape_[1] * input_shape_[2] * 32
    flattern_ = tf.reshape(rcl_layer_1_, [-1, input_shape_[1] * input_shape_[2] * 32])

    result_, linear_trans_ = softmax_layer(
        flattern_,
        flattern_dim_,
        output_shape_[1]
    )

    cross_entropy_ = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=linear_trans_, labels=output_placeholder_)
    )

    optimiser_ = tf.train.AdamOptimizer(learning_rate=learning_rate_).minimize(cross_entropy_)
    correct_prediction_ = tf.equal(tf.argmax(output_placeholder_, 1), tf.argmax(output_placeholder_, 1))
    accuracy_ = tf.reduce_mean(tf.cast(correct_prediction_, tf.float32))

    init_op_ = tf.global_variables_initializer()

    with tf.Session() as sess_:
        # initialise the variables
        sess_.run(init_op_)
        total_batch_ = int(training_data_.shape[0] / batch_size_)
        for epoch in range(epochs_):
            avg_cost_ = 0
            for i in range(total_batch_):
                batch_x_ = training_data_[i * batch_size_: (i + 1) * batch_size_]
                batch_y_ = training_labels_[i * batch_size_: (i + 1) * batch_size_]
                _, cost_ = sess_.run([optimiser_, cross_entropy_],
                                     feed_dict={
                                         input_placeholder_: batch_x_,
                                         output_placeholder_: batch_y_
                                     })
                avg_cost_ += cost_ / total_batch_
            test_acc_ = sess_.run(accuracy_,
                                  feed_dict={
                                      input_placeholder_: test_data_,
                                      output_placeholder_: test_labels_
                                  })
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost_),
                  "test accuracy: {:.3f}".format(test_acc_))

            print("\nTraining complete!")
            print(sess_.run(accuracy_,
                            feed_dict={
                                input_placeholder_: test_data_,
                                output_placeholder_: test_labels_
                            }
                            )
                  )
