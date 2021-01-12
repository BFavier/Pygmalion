#include <neural_networks/layers/layers.hpp>

using namespace pygmalion;
using namespace nlohmann;

ConvolutionLayer::ConvolutionLayer()
{
}

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayer& other)
{
    *this = other;
}

ConvolutionLayer::ConvolutionLayer(const json& dump)
{
    check_name(dump);
    const json& parameters = dump[type_name()];
    // Padding
    const json& padding_parameters = parameters["padding"];
    padding_size = padding_parameters["size"];
    padding_value = padding_parameters["value"];
    // Convolution
    const json& conv_parameters = parameters["convolution"];
    conv_weight = conv_parameters["weights"];
    conv_bias = conv_parameters["bias"];
    conv_stride = conv_parameters["stride"];
    // Batch normalization
    const json& batch_norm_parameters = parameters["batch_norm"];
    batch_norm_mean = batch_norm_parameters["running_mean"];
    batch_norm_variance = batch_norm_parameters["running_var"];
    batch_norm_weight = batch_norm_parameters["weight"];
    batch_norm_bias = batch_norm_parameters["bias"];
    // Pooling window
    const json& pooling_parameters = parameters["pooling"];
    pooling_window = pooling_parameters["window"];
    // Non linear
    non_linear = functions.at(parameters["non_linear"]);
}

ConvolutionLayer::~ConvolutionLayer()
{
}

std::string ConvolutionLayer::type_name() const
{
    return "ConvolutionLayer";
}

ConvolutionLayer& ConvolutionLayer::operator=(const ConvolutionLayer& other)
{
    padding_size = other.padding_size;
    padding_value = other.padding_value;
    conv_weight = other.conv_weight;
    conv_bias = other.conv_bias;
    conv_stride = other.conv_stride;
    batch_norm_mean = other.batch_norm_mean;
    batch_norm_variance = other.batch_norm_variance;
    batch_norm_weight = other.batch_norm_weight;
    batch_norm_bias = other.batch_norm_bias;
    pooling_window = other.pooling_window;
    non_linear = other.non_linear;
    return *this;
}

Tensor<double> ConvolutionLayer::operator()(const Tensor<double>& input) const
{
    Tensor<double>&& padded = pad(input, padding_value, padding_size[0], padding_size[1], padding_size[2], padding_size[3]);
    Tensor<double>&& convolved = convolve(padded, conv_weight, conv_bias, conv_stride[0], conv_stride[1]);
    Tensor<double>&& normalized = batch_normalize(convolved, batch_norm_mean, batch_norm_variance, batch_norm_weight, batch_norm_bias);
    Tensor<double>&& max_pooled = max_pool(normalized, pooling_window[0], pooling_window[1]);
    return non_linear(max_pooled);
}

UpsamplingLayer::UpsamplingLayer()
{
}

UpsamplingLayer::UpsamplingLayer(const UpsamplingLayer& other)
{
    (*this) = other;
}

UpsamplingLayer::UpsamplingLayer(const nlohmann::json& dump)
{
    check_name(dump);
    const json& parameters = dump[type_name()];
    // Padding
    const json& padding_parameters = parameters["padding"];
    padding_size = padding_parameters["size"];
    padding_value = padding_parameters["value"];
    // Convolution
    const json& conv_parameters = parameters["convolution"];
    conv_weight = conv_parameters["weights"];
    conv_bias = conv_parameters["bias"];
    conv_stride = conv_parameters["stride"];
    // Batch normalization
    const json& batch_norm_parameters = parameters["batch_norm"];
    batch_norm_mean = batch_norm_parameters["running_mean"];
    batch_norm_variance = batch_norm_parameters["running_var"];
    batch_norm_weight = batch_norm_parameters["weight"];
    batch_norm_bias = batch_norm_parameters["bias"];
    // Non linear
    non_linear = functions.at(parameters["non_linear"]);
}

UpsamplingLayer::~UpsamplingLayer()
{
}

std::string UpsamplingLayer::type_name() const
{
    return "UpsamplingLayer";
}

UpsamplingLayer& UpsamplingLayer::operator=(const UpsamplingLayer& other)
{
    padding_size = other.padding_size;
    padding_value = other.padding_value;
    conv_weight = other.conv_weight;
    conv_bias = other.conv_bias;
    conv_stride = other.conv_stride;
    batch_norm_mean = other.batch_norm_mean;
    batch_norm_variance = other.batch_norm_variance;
    batch_norm_weight = other.batch_norm_weight;
    batch_norm_bias = other.batch_norm_bias;
    non_linear = other.non_linear;
    return *this;
}

Tensor<double> UpsamplingLayer::operator()(const Tensor<double>& input, const Tensor<double>& concat) const
{
    unsigned int D = concat.shape.size();
    unsigned int width = concat.shape[D-1];
    unsigned int height = concat.shape[D-2];
    Tensor<double> output = resample(input, height, width);
    output = Tensor<double>::concatenate({output, concat}, 0);
    output = pad(output, padding_value, padding_size[0], padding_size[1], padding_size[2], padding_size[3]);
    output = convolve(output, conv_weight, conv_bias, conv_stride[0], conv_stride[1]);
    output = batch_normalize(output, batch_norm_mean, batch_norm_variance, batch_norm_weight, batch_norm_bias);
    return non_linear(output);
}

DenseLayer::DenseLayer()
{
}

DenseLayer::DenseLayer(const DenseLayer& other)
{
    *this = other;
}

DenseLayer::DenseLayer(const nlohmann::json& dump)
{
    check_name(dump);
    json parameters = dump[type_name()];
    // Linear
    json linear_parameters = parameters["linear"];
    weight = linear_parameters["weights"];
    bias = linear_parameters["bias"];
    // Non linear
    non_linear = functions.at(parameters["non_linear"]);
}

DenseLayer::~DenseLayer()
{
}

std::string DenseLayer::type_name() const
{
    return "DenseLayer";
}

DenseLayer& DenseLayer::operator=(const DenseLayer& other)
{
    weight = other.weight;
    bias = other.bias;
    non_linear = other.non_linear;
    return *this;
}

Tensor<double> DenseLayer::operator()(const Tensor<double>& input) const
{
    Tensor<double>&& weighted = linear(input, weight, bias);
    return non_linear(weighted);
}


//Python interface
extern "C" ConvolutionLayer* new_ConvolutionLayer(char* dump)
{
    json data = json::parse(dump);
    return new ConvolutionLayer(data);
}

extern "C" Tensor<double>* apply_ConvolutionLayer(ConvolutionLayer* layer, Tensor<double>* input)
{
    return new Tensor<double>((*layer)(*input));
}

extern "C" void del_ConvolutionLayer(ConvolutionLayer* pointer)
{
    delete pointer;
}

extern "C" UpsamplingLayer* new_UpsamplingLayer(char* dump)
{
    json data = json::parse(dump);
    return new UpsamplingLayer(data);
}

extern "C" Tensor<double>* apply_UpsamplingLayer(UpsamplingLayer* layer, Tensor<double>* input, Tensor<double>* concat)
{
    return new Tensor<double>((*layer)(*input, *concat));
}

extern "C" void del_UpsamplingLayer(UpsamplingLayer* pointer)
{
    delete pointer;
}

extern "C" DenseLayer* new_DenseLayer(char* dump)
{
    json data = json::parse(dump);
    return new DenseLayer(data);
}

extern "C" Tensor<double>* apply_DenseLayer(DenseLayer* layer, Tensor<double>* input)
{
    return new Tensor<double>((*layer)(*input));
}

extern "C" void del_DenseLayer(DenseLayer* pointer)
{
    delete pointer;
}
