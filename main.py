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
    parser.add_argument('-d', '--depth', type=int)
    parser.add_argument('-f', '--adaptive-learning-factor', type=float)
    parser.add_argument('-b', '--batch-size', type=int)

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
    do_grid_search = False
    train_args = None
    if len(argv) > 3:
        if argv[3] == 'grid-search':
            do_grid_search = True
            coarse = True
        else:
            train_args = parse_train_args(argv[3:])
            print train_args

    train_data_length_ = None
    test_data_length_ = None

    if do_grid_search:
        train_data_length_ = 2000
        test_data_length_ = 500
        np.random.seed(12345)

    dataset_name_ = argv[2]
    if dataset_name_.upper() == "MNIST":
        # Setting the parameters
        input_shape_ = [None, 28, 28, 1]
        output_shape_ = [None, 10]
        learning_rate_ = .0005
        epochs_ = 12
        batch_size_ = 1
        num_filter_ = 64
        buffer_size_ = 10000
        recurrent_depth_ = 3

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
        epochs_ = 25
        batch_size_ = 1
        num_filter_ = 96
        buffer_size_ = 10000
        recurrent_depth_ = 3

        if not use_grayscale:
            training_data_ = np.empty((0, 32, 32, 3))
        else:
            training_data_ = np.empty((0, 32, 32, 1))

        training_labels_ = np.empty((0, 10))
        for i in range(1, 6):
            training_data_batch_, training_labels_batch_ = data_loader(
                "cifar10",
                "data_batch_" + str(i),
                dtype=RCNN_tf.PRECISION_NP,
                use_grayscale=use_grayscale,
                data_length=int(train_data_length_ / 5)
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
        epochs_ = 25
        batch_size_ = 1
        num_filter_ = 96
        buffer_size_ = 10000
        recurrent_depth_ = 3

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

    if not do_grid_search:
        if train_args:
            if train_args.depth is not None:
                recurrent_depth_ = train_args.depth
            if train_args.adaptive_learning_factor:
                adaptive_learning_factor = train_args.adaptive_learning_factor
            if train_args.batch_size:
                batch_size_ = train_args.batch_size

        if recurrent_depth_ == 0:
            num_filter_test_ = 128
        else:
            num_filter_test_ = num_filter_

        # we increase batch size by a factor of k
        learning_rate_test = learning_rate_ * np.sqrt(batch_size_)

        print recurrent_depth_, adaptive_learning_factor, batch_size_

        rcnn = RCNN_tf.RCNN(
            input_shape=input_shape_,
            output_shape=output_shape_,
            processing_unit=processing_unit_,
            learning_rate=learning_rate_test,
            num_filter=num_filter_test_,
            shuf_buf_size=buffer_size_,
            recurrent_depth=recurrent_depth_
        )

        accuracy = rcnn.train(
            train_data_feats=training_data_,
            train_data_labels=training_labels_,
            val_data_feats=val_data_,
            val_data_labels=val_label_,
            test_data_feats=test_data_,
            test_data_labels=test_labels_,
            batch_size=batch_size_,
            epochs=epochs_,
            create_graph=False,
            print_vars=True,
            adaptive_learning_factor=adaptive_learning_factor,
            dir_name='final-{}-depth_{}-learningfactor_{}-batch_{}'.format(
                dataset_name_, recurrent_depth_,
                adaptive_learning_factor, batch_size_)
        )
        print 'accuracy {}'.format(accuracy)

        fname = 'final-accuracy-{}-{}.txt'.format(dataset_name_, recurrent_depth_, 'w+')
        final_accuracy = safe_open(fname)

        final_accuracy.write(
            'depth = {}, learningfactor = {}, batch {}: {}\n'.format(
                recurrent_depth_, adaptive_learning_factor,
                batch_size_, accuracy
            )
        )
        final_accuracy.close()

    else:
        fname = 'accuracies-{}.txt'.format(dataset_name_, 'w+')
        accuracies = safe_open(fname)
        epochs_ = 10
        for recurrent_depth_ in [0, 3, 6]:
            for adaptive_learning_factor in [0.01, 0.1, 1]:
                for batch_size_ in [1, 32, 100]:
                    tf.reset_default_graph()
                    # as per the paper to set the number of weights to be roughly
                    # the same
                    print 'Training with depth {} and learning factor {}'.format(
                        recurrent_depth_, adaptive_learning_factor
                    )

                    if recurrent_depth_ == 0:
                        num_filter_test_ = 128
                    else:
                        num_filter_test_ = num_filter_

                    # we increase batch size by a factor of k
                    learning_rate_test = learning_rate_ * np.sqrt(batch_size_)

                    rcnn = RCNN_tf.RCNN(
                        input_shape=input_shape_,
                        output_shape=output_shape_,
                        processing_unit=processing_unit_,
                        learning_rate=learning_rate_test,
                        num_filter=num_filter_test_,
                        shuf_buf_size=buffer_size_,
                        recurrent_depth=recurrent_depth_
                    )

                    accuracy = rcnn.train(
                        train_data_feats=training_data_,
                        train_data_labels=training_labels_,
                        val_data_feats=val_data_,
                        val_data_labels=val_label_,
                        test_data_feats=test_data_,
                        test_data_labels=test_labels_,
                        batch_size=batch_size_,
                        epochs=epochs_,
                        create_graph=False,
                        print_vars=True,
                        adaptive_learning_factor=adaptive_learning_factor,
                        dir_name='{}-depth_{}-learningfactor_{}-batch_{}'.format(
                            dataset_name_, recurrent_depth_,
                            adaptive_learning_factor, batch_size_)
                    )
                    print 'accuracy {}'.format(accuracy)

                    accuracies.write(
                        'depth = {}, learningfactor = {}, batch {}: {}\n'.format(
                            recurrent_depth_, adaptive_learning_factor,
                            batch_size_, accuracy
                        )
                    )
                    # in case we're interrupted we'll save our progress somewhat
                    accuracies.flush()
        accuracies.close()


if __name__ == '__main__':
    main(sys.argv)
