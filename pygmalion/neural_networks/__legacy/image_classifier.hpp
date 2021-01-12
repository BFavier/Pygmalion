#pragma once
#include <neural_networks/layers/layers.hpp>
#include <image/image.hpp>
#include <fstream>

namespace pygmalion
{
    class ImageClassifier : public Model
    {
    public:
        ImageClassifier();
        ImageClassifier(const ImageClassifier& other);
        ImageClassifier(const std::string& path);
        ~ImageClassifier();
    public:
        std::vector<std::string> categories;
        std::vector<unsigned int> input_shape;
        Tensor<double> mean;
        Tensor<double> standard_deviation;
        std::vector<ConvolutionLayer> convolution_layers;
        std::vector<DenseLayer> dense_layers;
    public:
        virtual std::string type_name() const;
        unsigned int predict_index(const Image& image) const;
        std::string predict(const Image& image) const;
    public:
        ImageClassifier& operator=(const ImageClassifier& other);
        Tensor<double> operator()(const Tensor<double>& input) const;
    protected:
        void parse(const nlohmann::json& dump);
    };

    //Python interface
    extern "C" ImageClassifier* new_ImageClassifier(char* path);
    extern "C" Tensor<double>* apply_ImageClassifier(ImageClassifier* classifier, Tensor<double>* input);
    extern "C" void del_ImageClassifier(ImageClassifier* classifier);
    // extern "C" unsigned int ImageClassifier_n_categories(ImageClassifier* classifier);
    // extern "C" char** ImageClassifier_categories(ImageClassifier* classifier);
}
