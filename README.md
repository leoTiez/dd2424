# Deep Learning Project
## Recurrent Convolutional Neural Network for Object Recognition

### Requirements
- Pyhton 2.7
- NumPy
- TensorFlow

It is recommended to use the `requirements.txt` file for the installation. Run

```bash
pip install -r requirements.txt --user
```

### Data
To download the data simply run

```bash
bash download-data.sh
```
Data sets are ony downloaded if the have not been downloaded already.
Please add new data sets to the download data file.

Also, it is necessary to convert the data properly before using it with tensorflow. 
In case of a new data set please provide next to a loading function a data pre-processing
function which converts the data set to a 4-dim numpy array in `(N, H, W, C)` format
meaning
- N = number of images
- H = height of the image
- W = width of the image
- C = number of channels

### Usage
For an easy usage a class implementation is provided, setting up the architecture and
the necessary class variables. It can be initialized using

```python
rcnn = RCNN_tf.RCNN(
        input_shape=input_shape_, # shape of the input images
        output_shape=output_shape_, # shape of the output 
        processing_unit=processing_unit_, # device responsible for the computation, e.g. CPU or GPU
        learning_rate=learning_rate_, # initial learning rate value
        num_filter=num_filter_, # number of feature maps per convolutional filter
        shuf_buf_size=buffer_size_, # size of the data shuffle buffer 
    )
```

To train the network run 

```python
accuracy = rcnn.train(
        train_data_feats=training_data_, # training data
        train_data_labels=training_labels_, # training labels
        val_data_feats=val_data_, # validation data
        val_data_labels=val_label_, # validation labels
        test_data_feats=test_data_, # test data
        test_data_labels=test_labels_, # test labels
        batch_size=batch_size_, # batch size
        training_depth=depth_learning_, # number of recurrent iterations per RCL while learning
        test_depth=depth_test_, # number of recurrent iterations per RCL while testing
        epochs=epochs_, # number of epochs
        create_graph=False, # flag for creating tensorflow graph
        print_vars=True, # flag for printing variable dimensions
        adaptive_learning_factor=adaptive_learning_factor_, # annealing factor for the learning rate
        dir_name='final-{}-depth_{}-learningfactor_{}-batch_{}'.format(
            dataset_name_, depth_learning_,
            adaptive_learning_factor_, batch_size_) # directory for saving the output
    )
```

### Execution
The main function in `main.py` provides an example how to initialize and use the 
class `RCNN`. The main file can be executed using the following command

```bash
python main.py "${device}" "${dataset}" -dl "${depth_learning}" -dt "${depth_test}" -f "${annealing_rate}" -b "${batch_size}" \
            -lls "${learning_data_set_size}" -lts "${testing_data_set_size}" -nf 64 -e "${epochs}"
```

where
- device: gpu and cpu
- dataset: name of the data set - mnist, cifar10 or cifar100
- dl: number of recurrent iterations while learning 
- dt: number of recurrent iterations while testing
- f: annealing factor
- b: batch size
- lls: data set size for the training phase. If nothing is passed the whole data set is used
- lts: data set size for the testing phase. If nothing is passed the whole data set is used
- nf: number of filters per convolutional layer
- e: number of epochs

There is one example file for performing a grid search:
```bash
bash grid-search-all.sh
```

