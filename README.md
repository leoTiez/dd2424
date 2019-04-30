# Deep Learning Project
## Recurrent Convolutional Neural Network for Object Recognition 

### Pre requirements
Please install:
- Pyhton 2.7
- NumPy
- TensorFlow
- cPickle

### Data
To download the data simply run

```bash
bash download-data.sh
```
Data sets are ony downloaded if the have not been downloaded already.
Please add new data sets to the download data file.

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


