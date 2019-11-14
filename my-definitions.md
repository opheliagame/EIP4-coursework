1. **Convolution**: (literally) twisting, molding around something. 
                (contexually) process of kernel twisting, molding(convolving) around the layer's input. 
                 How the filter convolves is defined by the its' size and stride 

2. **Filters, Kernels**: 2D or 3D matrices that are learnt by the network as feature detectors through the training process

3. **Epochs**: Number of times the network has seen the entire training set

4. **1X1 Convolution**: a convolution with kernel size 1X1. Used to reduce the z-depth(number of kernels) of the output, helps in controlling the number of trainable parameters.
                    Used in the transition block after max pooling and before 3X3 convolutions.

5. **3X3 Convolution**: a convolution with kernel size 3X3. The most popular convolution. 

6. **Feature Maps**: Output of a convolution by a kernel, has high values for the feature being detected by that kernel 

7. **Activation Function**: a function that maps linear input space to non-linear output space

8. **Receptive Field**: Size of the input layer that can be seen by the kernel at a time. Can be local, or global. 
