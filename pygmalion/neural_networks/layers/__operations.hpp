#pragma once
#include <tensor/tensor.hpp>
#include <map>
#include <cmath>
#include <algorithm>

namespace pygmalion
{
    ///< Applies the normalization to the input tensor. The input tensor must be of shape {Channels, ...} and the mean and standard_deviation of shape {Channels}.
    Tensor<double> normalize(const Tensor<double>& input, const Tensor<double>& mean, const Tensor<double>& standard_deviation);
    ///< Returns the tensor padded with the given value. The tensor must be of shape {Channels, Height, Width}. The returned tensor is of shape {Channels, Height+top+bottom, Width+left+right}.
    Tensor<double> pad(const Tensor<double>& input, double value, unsigned int left, unsigned int right, unsigned int top, unsigned int bottom);
    ///< Unroll an input tensor of shape {C, H, W} into a shape {H_out, W_out, C, H_kernel, W_kernel}. This is used for faster convolutions.
    Tensor<double> unroll(const Tensor<double>& input, unsigned int H_kernel, unsigned int W_kernel, unsigned int S_h, unsigned int S_w);
    ///< Returns the tensor convolved by the kernel, with the given stride. The input must have a shape of {in_channels, Height, Width}, the kernel must have a shape of {C_out, C_in, Kernel_height, kernel_Width}, the bias must have a shape of {C_out}.
    Tensor<double> convolve(const Tensor<double>& input, const Tensor<double>& kernel, const Tensor<double>& bias, unsigned int stride_height, unsigned int stride_width);
    ///< Performs a max pooling operation on the input. The input must be of shape {Channels, Height, Width}.
    Tensor<double> max_pool(const Tensor<double>& input, unsigned int height, unsigned int width);
    ///< Performs batch normalization on the input tensor. The input must be of shape {Channels, Height, Width}. The mean, variance, weight and bias must be of shape {Channels}.
    Tensor<double> batch_normalize(const Tensor<double>& input, const Tensor<double>& mean, const Tensor<double>& variance, const Tensor<double>& weight, const Tensor<double>& bias);
    ///< Apply a linear dense layer to the input. The input must be of shape {N_in, ...}, the weights must be of shape {N_out, N_in}, the bias must be of shape {N_out}.
    Tensor<double> linear(const Tensor<double>& input, const Tensor<double>& weights, const Tensor<double>& bias);
    ///< Resample a tensor representing an image/feature map using bilinear interpolation. The input tensor must be of shape {Channels, Height, Width}.
    Tensor<double> resample(const Tensor<double>& input, unsigned int new_height, unsigned int new_width);
    ///< Defining the prototype of the non linear functions
    typedef Tensor<double> (*TensorOperation)(const Tensor<double>&);
    ///< Apply the identity function to the input: f(x) = x
    Tensor<double> identity(const Tensor<double>& input);
    ///< Apply the ReLU function to the input: f(x) = max(0., x)
    Tensor<double> relu(const Tensor<double>& input);
    ///< Apply the tanh function to the input
    Tensor<double> tanh(const Tensor<double>& input);

    ///< Dictionary of available tensor operations
    extern const std::map<std::string, TensorOperation> functions;

    //python interface
    extern "C" Tensor<double>* normalize(Tensor<double>* input, Tensor<double>* mean, Tensor<double>* standard_deviation);
    extern "C" Tensor<double>* pad(Tensor<double>* input, double value, unsigned int left, unsigned int right, unsigned int top, unsigned int bottom);
    extern "C" Tensor<double>* convolve(Tensor<double>* input, Tensor<double>* kernel, Tensor<double>* bias, unsigned int stride_height, unsigned int stride_width);
    extern "C" Tensor<double>* max_pool(Tensor<double>* input, unsigned int height, unsigned int width);
    extern "C" Tensor<double>* batch_normalize(Tensor<double>* input, Tensor<double>* mean, Tensor<double>* variance, Tensor<double>* weight, Tensor<double>* bias);
    extern "C" Tensor<double>* linear(Tensor<double>* input, Tensor<double>* weights, Tensor<double>* bias);
    extern "C" Tensor<double>* resample(Tensor<double>* input, unsigned int new_height, unsigned int new_width);
    extern "C" Tensor<double>* Tensor_relu(Tensor<double>* input);
    extern "C" Tensor<double>* Tensor_tanh(Tensor<double>* input);
}
