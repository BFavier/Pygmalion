#include <neural_networks/image_classifier.hpp>

using namespace pygmalion;
using namespace nlohmann;

ImageClassifier::ImageClassifier()
{
}

ImageClassifier::ImageClassifier(const ImageClassifier& other)
{
    *this = other;
}

ImageClassifier::ImageClassifier(const std::string& path)
{
    std::ifstream file(path);
    json dump = json::parse(file);
    parse(dump);
}

ImageClassifier::~ImageClassifier()
{
}

std::string ImageClassifier::type_name() const
{
    return "ImageClassifier";
}

unsigned int ImageClassifier::predict_index(const Image& image) const
{
    Tensor<double> input = image.as_tensor();
    Tensor<double> T((*this)(input));
    double* I = input.address();
    double activation = -std::numeric_limits<double>::infinity();
    unsigned int index = 0;
    for (unsigned int i=0; i<T.size; i++)
    {
        if (I[i] > activation)
        {
            activation = I[i];
            index = i;
        }
    }
    return index;
}

std::string ImageClassifier::predict(const Image& image) const
{
    Tensor<double> input = image.as_tensor();
    if (input.shape != input_shape)
    {
        throw("Input tensor doesn't have the right shape");
    }
    unsigned int i = predict_index(input);
    return categories[i];
}

ImageClassifier& ImageClassifier::operator=(const ImageClassifier& other)
{
    categories = other.categories;
    input_shape = other.input_shape;
    mean = other.mean;
    standard_deviation = other.standard_deviation;
    convolution_layers = other.convolution_layers;
    dense_layers = other.dense_layers;
    return *this;
}

Tensor<double> ImageClassifier::operator()(const Tensor<double>& input) const
{
    Tensor<double> T(normalize(input, mean, standard_deviation));
    for (ConvolutionLayer conv : convolution_layers)
    {
        T = conv(T);
    }
    for (DenseLayer dense : dense_layers)
    {
        T = dense(T);
    }
    return T;
}

void ImageClassifier::parse(const json& dump)
{
    check_name(dump);
    check_version(dump);
    json parameters = dump[type_name()];
    categories = static_cast<std::vector<std::string>>(parameters["categories"]);
    input_shape = static_cast<std::vector<unsigned int>>(parameters["input_shape"]);
    mean = Tensor<double>(parameters["mean"]).flatten();
    standard_deviation = Tensor<double>(parameters["std"]).flatten();
    for (const json& conv_dump : parameters["convolution_layers"])
    {
        convolution_layers.push_back(conv_dump);
    }
    for (const json& dense_dump : parameters["dense_layers"])
    {
        dense_layers.push_back(dense_dump);
    }
}

//Python interface
extern "C" ImageClassifier* new_ImageClassifier(char* path)
{
    return new ImageClassifier(std::string(path));
}

extern "C" Tensor<double>* apply_ImageClassifier(ImageClassifier* classifier, Tensor<double>* input)
{
    return new Tensor<double>((*classifier)(*input));
}

extern "C" void del_ImageClassifier(ImageClassifier* classifier)
{
    delete classifier;
}
