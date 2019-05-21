import sys
import RCNN_tf
from load_data import data_loader
import numpy as np
from os import path
import tensorflow as tf
from argparse import ArgumentParser

VAL_SET_PERCENT_LEN = 10


def parse_train_args(args):
    parser = ArgumentParser()
    parser.add_argument('-dl', '--depth-learning', type=int)
    parser.add_argument('-dt', '--depth-test', type=int)
    parser.add_argument('-f', '--adaptive-learning-factor', type=float)
    parser.add_argument('-b', '--batch-size', type=int)
    parser.add_argument('-lls', '--length-learning-set', type=int)
    parser.add_argument('-lts', '--length-testing-set', type=int)
    parser.add_argument('-nf', '--number-filter', type=int)
    parser.add_argument('-e', '--epochs', type=int)


    return parser.parse_args(args)


def safe_open(fname):
    if path.isfile(fname):
        raise Exception(
            '{} already exists. Do not want to overwrite. Exiting'.format(
                fname))
    return open(fname, 'w+')


def main(argv):
    print argv
    processing_unit_ = argv[1]
    if processing_unit_.upper() == "CPU" or processing_unit_ is None:
        processing_unit_ = "/cpu:0"
    elif processing_unit_.upper() == "GPU":
        processing_unit_ = "/gpu:0"
    else:
        raise ValueError("Device type is not supported. Use either GPU or CPU")

    coarse = False
    train_args = None
    if len(argv) > 3:
        train_args = parse_train_args(argv[3:])

    depth_learning_ = 3
    depth_test_ = 5
    adaptive_learning_factor_ = 0.01
    batch_size_ = 1
    train_data_length_ = None
    test_data_length_ = None
    num_filter_ = 96
    epochs_ = None
    if train_args:
        if train_args.depth_learning is not None:
            depth_learning_ = train_args.depth_learning
        if train_args.depth_test is not None:
            depth_test_ = train_args.depth_test
        if train_args.adaptive_learning_factor is not None:
            adaptive_learning_factor_ = train_args.adaptive_learning_factor
        if train_args.batch_size is not None:
            batch_size_ = train_args.batch_size
        if train_args.length_learning_set is not None:
            train_data_length_ = train_args.length_learning_set
        if train_args.length_testing_set is not None:
            test_data_length_ = train_args.length_testing_set
        if train_args.number_filter is not None:
            num_filter_ = train_args.number_filter
        if train_args.epochs is not None:
            epochs_ = train_args.epochs

    dataset_name_ = argv[2]
    if dataset_name_.upper() == "MNIST":
        # Setting the parameters
        input_shape_ = [None, 28, 28, 1]
        output_shape_ = [None, 10]
        learning_rate_ = .0005
        if epochs_ is not None:
            epochs_ = 12
        buffer_size_ = 10000

        training_data_, training_labels_ = data_loader(
            "mnist",
            "train",
            dtype=RCNN_tf.PRECISION_NP,
            data_length=train_data_length_
        )

        val_data_ = training_data_[
                    -training_data_.shape[0] / VAL_SET_PERCENT_LEN:]
        val_label_ = training_labels_[
                     -training_labels_.shape[0] / VAL_SET_PERCENT_LEN:]

        training_data_ = training_data_[
                         :-training_data_.shape[0] / VAL_SET_PERCENT_LEN]
        training_labels_ = training_labels_[
                           :-training_labels_.shape[0] / VAL_SET_PERCENT_LEN]

        test_data_, test_labels_ = data_loader(
            "mnist",
            "test",
            dtype=RCNN_tf.PRECISION_NP,
            data_length=test_data_length_
        )

    elif dataset_name_.upper() == "CIFAR10":
        # Setting the parameters
        use_grayscale = False
        if not use_grayscale:
            input_shape_ = [None, 32, 32, 3]
        else:
            input_shape_ = [None, 32, 32, 1]

        output_shape_ = [None, 10]
        learning_rate_ = .0005
        if epochs_ is not None:
            epochs_ = 25
        buffer_size_ = 10000

        if not use_grayscale:
            training_data_ = np.empty((0, 32, 32, 3))
        else:
            training_data_ = np.empty((0, 32, 32, 1))

        training_labels_ = np.empty((0, 10))
        for i in range(1, 6):
            if train_data_length_ is not None:
                train_data_length_ = int(train_data_length_ / 5)
            training_data_batch_, training_labels_batch_ = data_loader(
                "cifar10",
                "data_batch_" + str(i),
                dtype=RCNN_tf.PRECISION_NP,
                use_grayscale=use_grayscale,
                data_length=train_data_length_
            )

            training_data_ = np.concatenate((training_data_,
                                             training_data_batch_))
            training_labels_ = np.concatenate((training_labels_,
                                               training_labels_batch_))

        val_data_ = training_data_[
                    -training_data_.shape[0] / VAL_SET_PERCENT_LEN:]
        val_label_ = training_labels_[
                     -training_labels_.shape[0] / VAL_SET_PERCENT_LEN:]

        training_data_ = training_data_[
                         :-training_data_.shape[0] / VAL_SET_PERCENT_LEN]
        training_labels_ = training_labels_[
                           :-training_labels_.shape[0] / VAL_SET_PERCENT_LEN]

        test_data_, test_labels_ = data_loader(
            "cifar10",
            "test_batch",
            dtype=RCNN_tf.PRECISION_NP,
            use_grayscale=use_grayscale,
            data_length=test_data_length_
        )

    elif dataset_name_.upper() == "CIFAR100":
        # Setting the parameters
        use_grayscale = False
        if not use_grayscale:
            input_shape_ = [None, 32, 32, 3]
        else:
            input_shape_ = [None, 32, 32, 1]

        if coarse:
            output_shape_ = [None, 20]
        else:
            output_shape_ = [None, 100]

        learning_rate_ = .0001
        if epochs_ is not None:
            epochs_ = 25
        buffer_size_ = 10000

        training_data_, training_labels_ = data_loader(
            "cifar100",
            "train",
            dtype=RCNN_tf.PRECISION_NP,
            use_grayscale=use_grayscale,
            data_length=train_data_length_,
            coarse=coarse
        )

        val_data_ = training_data_[
                    -training_data_.shape[0] / VAL_SET_PERCENT_LEN:]
        val_label_ = training_labels_[
                     -training_labels_.shape[0] / VAL_SET_PERCENT_LEN:]

        training_data_ = training_data_[
                         :-training_data_.shape[0] / VAL_SET_PERCENT_LEN]
        training_labels_ = training_labels_[
                           :-training_labels_.shape[0] / VAL_SET_PERCENT_LEN]

        test_data_, test_labels_ = data_loader(
            "cifar100",
            "test",
            dtype=RCNN_tf.PRECISION_NP,
            use_grayscale=use_grayscale,
            data_length=test_data_length_,
            coarse=coarse
        )

    else:
        raise Exception(
            "dataset has to be one of 'mnist', 'cifar10' or 'cifar100'.")


    # we increase batch size by a factor of k
    learning_rate_to_batch_size_ = learning_rate_ * np.sqrt(batch_size_)

    rcnn = RCNN_tf.RCNN(
        input_shape=input_shape_,
        output_shape=output_shape_,
        processing_unit=processing_unit_,
        learning_rate=learning_rate_to_batch_size_,
        num_filter=num_filter_,
        shuf_buf_size=buffer_size_,
    )

    accuracy = rcnn.train(
        train_data_feats=training_data_,
        train_data_labels=training_labels_,
        val_data_feats=val_data_,
        val_data_labels=val_label_,
        test_data_feats=test_data_,
        test_data_labels=test_labels_,
        batch_size=batch_size_,
        training_depth=depth_learning_,
        test_depth=depth_test_,
        epochs=epochs_,
        create_graph=False,
        print_vars=True,
        adaptive_learning_factor=adaptive_learning_factor_,
        dir_name='final-{}-depth_{}-learningfactor_{}-batch_{}'.format(
            dataset_name_, depth_learning_,
            adaptive_learning_factor_, batch_size_)
    )
    print 'accuracy {}'.format(accuracy)

    fname = 'final-accuracy-{}-{}.txt'.format(dataset_name_, depth_learning_, 'w+')
    final_accuracy = safe_open(fname)

    final_accuracy.write(
        'depth = {}, learningfactor = {}, batch {}: {}\n'.format(
            depth_learning_, adaptive_learning_factor_,
            batch_size_, accuracy
        )
    )
    final_accuracy.close()


if __name__ == '__main__':
    main(sys.argv)
