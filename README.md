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
The `RecurrentConvolutionalLayer` inherits from the base layer class defined in
tensorflow keras. Thus, it can be initialized using similar parameters.
One example is

```python
RecurrentConvolutionalLayer(
    32, # Number of parameters
    (3, 3), # Filter size
    execution_depth=3, # Number of recurrent iterations
    input_shape=(32, 32, 3) # Input size
)
```

Please remember to adapt the input size when using a new data set.
The layer can be added to existing network definitions, or can be
stacked together with other layers.

For easy usage a class implementation is provided, setting up the architecture and
the necessary class variables. It can be initialized using

```python
rcnn = RCNN(
        input_shape, # file shape
        device_name="/cpu:0",   # defining the device which should be used
                                # Use "/gpu:0" if you want to use the gpu
                                # implementation instead
        output_shape=[None, 10], # output shape. Second value is the number of classes
        learning_rate=.1,
        num_filter=64,  # Number of filters used for the recurrent and the normal
                        # convolutional layer
        shuffle_buffer_size=10000, 
        conv_layer_filter_shape=[5, 5], # filter shape of the normal conv layer
        rec_conv_layer_filter_shape=[3, 3], # filter shape of the recurrent conv layers
        pooling_shape=[3, 3], 
        pooling_stride_shape=[2, 2]
    )
```

To train the network run 

```python
rcnn.train(
            training_data_features,
            training_data_labels,
            val_data_features,
            val_data_labels,
            batch_size=100,
            epochs=7,
            create_graph=True
    )
```

### Execution
The main function in `RCNN_tf.py` provides an example how to initialize and use the 
class `RCNN`. All main files should accept an additional command line parameter
defining the execution mode (cpu or gpu). Thus, it should be possible to run 
the main files using the command

```bash
python main_file.py cpu
```

or 

```bash
python main_file.py gpu
```

