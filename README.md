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


