#include <neural_networks/image_segmenter.hpp>

using namespace pygmalion;
using namespace nlohmann;

ImageSegmenter::ImageSegmenter()
{
}

ImageSegmenter::ImageSegmenter(const ImageSegmenter& other)
{
    *this = other;
}

ImageSegmenter::ImageSegmenter(const std::string& path)
{
    std::ifstream file(path);
    json dump = json::parse(file);
    parse(dump);
}

ImageSegmenter::~ImageSegmenter()
{
}

std::string ImageSegmenter::type_name() const
{
    return "ImageSegmenter";
}

Image ImageSegmenter::predict(const Image& image) const
{
    if (image.channels() != in_channels)
    {
        throw std::runtime_error(std::string(__func__)+": Unexpected number of input channels in the image.");
    }
    Tensor<double> input = image.as_tensor();
    Tensor<double> output = (*this)(input);
    Tensor<unsigned char> indexed = classes.at_index(output.index_max());
    return Image(indexed);
}

ImageSegmenter& ImageSegmenter::operator=(const ImageSegmenter& other)
{
    classes = other.classes;
    in_channels = other.in_channels;
    mean = other.mean;
    standard_deviation = other.standard_deviation;
    convolution_layers = other.convolution_layers;
    upsampling_layers = other.upsampling_layers;
    out_layer = other.out_layer;
    return *this;
}

Tensor<double> ImageSegmenter::operator()(const Tensor<double>& input) const
{
    Tensor<double> output = normalize(input, mean, standard_deviation);
    std::list<Tensor<double>> down;
    for (ConvolutionLayer conv : convolution_layers)
    {
        down.push_back(output);
        output = conv(output);
    }
    for (UpsamplingLayer upsampling : upsampling_layers)
    {
        output = upsampling(output, down.back());
        down.pop_back();
    }
    return out_layer(output);
}

void ImageSegmenter::parse(const json& dump)
{
    check_name(dump);
    check_version(dump);
    json parameters = dump[type_name()];
    classes = parameters["classes"];
    mean = Tensor<double>(parameters["mean"]).flatten();
    standard_deviation = Tensor<double>(parameters["std"]).flatten();
    in_channels = parameters["in_channels"];
    for (const json& conv_dump : parameters["convolution_layers"])
    {
        convolution_layers.push_back(conv_dump);
    }
    for (const json& upsampling_dump : parameters["upsampling_layers"])
    {
        upsampling_layers.push_back(upsampling_dump);
    }
    out_layer = parameters["out_layer"];
}

//Python interface
extern "C" ImageSegmenter* new_ImageSegmenter(char* path)
{
    return new ImageSegmenter(std::string(path));
}

extern "C" Tensor<double>* apply_ImageSegmenter(ImageSegmenter* classifier, Tensor<double>* input)
{
    return new Tensor<double>((*classifier)(*input));
}

extern "C" void del_ImageSegmenter(ImageSegmenter* classifier)
{
    delete classifier;
}
