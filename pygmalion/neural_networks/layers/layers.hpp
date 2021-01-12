#pragma once
#include <tensor/operations.hpp>
#include <templates/model.hpp>

namespace pygmalion
{
    ///< A class performing padding/convolution/batch normalization/max pooling/gating on the input image data
    class ConvolutionLayer : public Model
    {
    public:
        ConvolutionLayer();
        ConvolutionLayer(const ConvolutionLayer& other);
        ConvolutionLayer(const nlohmann::json& dump);
        ~ConvolutionLayer();
    public:
        std::array<unsigned int, 4> padding_size;
        double padding_value;
        Tensor<double> conv_weight;
        Tensor<double> conv_bias;
        std::array<unsigned int, 2> conv_stride;
        Tensor<double> batch_norm_mean;
        Tensor<double> batch_norm_variance;
        Tensor<double> batch_norm_weight;
        Tensor<double> batch_norm_bias;
        std::array<unsigned int, 2> pooling_window;
        TensorOperation non_linear;
    public:
        virtual std::string type_name() const;
    public:
        ConvolutionLayer& operator=(const ConvolutionLayer& other);
        Tensor<double> operator()(const Tensor<double>& input) const;
    };

    ///< A class performing upsampling on the input image data
    class UpsamplingLayer : public Model
    {
    public:
        UpsamplingLayer();
        UpsamplingLayer(const UpsamplingLayer& other);
        UpsamplingLayer(const nlohmann::json& dump);
        ~UpsamplingLayer();
    public:
        std::array<unsigned int, 4> padding_size;
        double padding_value;
        Tensor<double> conv_weight;
        Tensor<double> conv_bias;
        std::array<unsigned int, 2> conv_stride;
        Tensor<double> batch_norm_mean;
        Tensor<double> batch_norm_variance;
        Tensor<double> batch_norm_weight;
        Tensor<double> batch_norm_bias;
        TensorOperation non_linear;
    public:
        virtual std::string type_name() const;
    public:
        UpsamplingLayer& operator=(const UpsamplingLayer& other);
        Tensor<double> operator()(const Tensor<double>& input, const Tensor<double>& concat) const;
    };

    ///< A class performing linear operation/gating on the input tensor data
    class DenseLayer : public Model
    {
    public:
        DenseLayer();
        DenseLayer(const DenseLayer& other);
        DenseLayer(const nlohmann::json& dump);
        ~DenseLayer();
    public:
        Tensor<double> weight;
        Tensor<double> bias;
        TensorOperation non_linear;
    public:
        virtual std::string type_name() const;
    public:
        DenseLayer& operator=(const DenseLayer& other);
        Tensor<double> operator()(const Tensor<double>& input) const;
    };

    //Python interface
    extern "C" ConvolutionLayer* new_ConvolutionLayer(char* dump);
    extern "C" Tensor<double>* apply_ConvolutionLayer(ConvolutionLayer* layer, Tensor<double>* input);
    extern "C" void del_ConvolutionLayer(ConvolutionLayer* pointer);

    extern "C" UpsamplingLayer* new_UpsamplingLayer(char* dump);
    extern "C" Tensor<double>* apply_UpsamplingLayer(UpsamplingLayer* layer, Tensor<double>* input, Tensor<double>* concat);
    extern "C" void del_UpsamplingLayer(UpsamplingLayer* pointer);

    extern "C" DenseLayer* new_DenseLayer(char* dump);
    extern "C" Tensor<double>* apply_DenseLayer(DenseLayer* layer, Tensor<double>* input);
    extern "C" void del_DenseLayer(DenseLayer* pointer);
}
