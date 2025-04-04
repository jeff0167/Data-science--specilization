### Import and determining error/warning level###
## file produced by Daniel van Dijk Jacobsen after the the paper by Ronneberger et al

import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow
from tensorflow import keras

from keras import Input, Model
from keras.layers import BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Dropout, MaxPooling2D, ReLU
from keras.utils import plot_model

### Checking Environment ###

print("Num GPUs Available: ", len(tensorflow.config.list_physical_devices('GPU')))

### Data Generation ###



### Model Architecture ####

UNetModel_input = Input(shape = (512, 512, 1), name = 'Input')

# Encoding

def encoding_block(input_tensor, block_name, filter_size):
    x = Conv2D(filters = filter_size, kernel_size = (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', name = block_name + 'Conv1') (input_tensor)
    x = Dropout(rate = 0.1, name = block_name + 'Drop') (x)
    shortcut = Conv2D(filters = filter_size, kernel_size = (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', name = block_name + 'Conv2') (x)
    x = BatchNormalization(name = block_name + 'BatchNorm') (shortcut)
    x = ReLU(name = block_name + 'ReLU') (x)
    x = MaxPooling2D(pool_size = (2, 2), name = block_name + 'MaxPool') (x)

    return x, shortcut

x, shortcut1 = encoding_block(UNetModel_input, 'Encoding1', 16)
x, shortcut2 = encoding_block(x, 'Encoding2', 32)
x, shortcut3 = encoding_block(x, 'Encoding3', 64)
x, shortcut4 = encoding_block(x, 'Encoding4', 128)

# Bottleneck

x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', name = 'BottleneckConv1') (x)
x = BatchNormalization(name = 'BottleneckBatchNorm') (x)
x = ReLU(name = 'BottleneckReLU') (x)
x = Dropout(rate = 0.3) (x)
x = Conv2D(filters = 256, kernel_size = (3, 3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same', name = 'BottleneckConv2') (x)

# Decoding

def decoding_block(input_tensor, shortcut_tensor, block_name, filter_size):
    x = Conv2DTranspose(filters = filter_size, kernel_size = (2, 2), strides = (2, 2), padding = 'same', name = block_name + 'ConvTranspose') (input_tensor)
    x = concatenate([x, shortcut_tensor], name = block_name + 'Concatenate')
    x = BatchNormalization(name = block_name + 'BatchNorm') (x)
    x = ReLU(name = block_name + 'ReLU') (x)

    return x

x = decoding_block(x, shortcut4, 'Decoding1', 128)
x = decoding_block(x, shortcut3, 'Decoding2', 64)
x = decoding_block(x, shortcut2, 'Decoding3', 32)
x = decoding_block(x, shortcut1, 'Decoding4', 16)

UNetModel_output = Conv2D(1, (1, 1), activation='sigmoid', name = 'Output')(x)

UNetModel = Model(UNetModel_input, UNetModel_output, name = 'UNetModel')
UNetModel.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = 'accuracy')
plot_model(UNetModel, "rawModels/modelUnet.png")