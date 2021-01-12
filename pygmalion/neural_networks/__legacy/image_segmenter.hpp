#pragma once
#include <neural_networks/layers/layers.hpp>
#include <image/image.hpp>
#include <fstream>

namespace pygmalion
{
    class ImageSegmenter : public Model
    {
    public:
        ImageSegmenter();
        ImageSegmenter(const ImageSegmenter& other);
        ImageSegmenter(const std::string& path);
        ~ImageSegmenter();
    public:
        Tensor<unsigned char> categories;
        unsigned int channels_in;
        Tensor<double> mean;
        Tensor<double> standard_deviation;
        std::vector<ConvolutionLayer> convolution_layers;
        std::vector<UpsamplingLayer> upsampling_layers;
        ConvolutionLayer out_layer;
    public:
        virtual std::string type_name() const;
        Image predict(const Image& image) const;
    public:
        ImageSegmenter& operator=(const ImageSegmenter& other);
        Tensor<double> operator()(const Tensor<double>& input) const;
    protected:
        void parse(const nlohmann::json& dump);
    };

    //Python interface
    extern "C" ImageSegmenter* new_ImageSegmenter(char* path);
    extern "C" Tensor<double>* apply_ImageSegmenter(ImageSegmenter* classifier, Tensor<double>* input);
    extern "C" void del_ImageSegmenter(ImageSegmenter* classifier);
}