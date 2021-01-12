#pragma once
#include <tensor/tensor.hpp>
#include <string>
#include <cctype>
#include <vector>

namespace pygmalion
{
    class Image : protected Tensor<unsigned char>
    {
    public:
        Image();
        Image(const Image& other);
        Image(unsigned int height, unsigned int width, unsigned int channels);
        Image(std::vector<unsigned char> data, unsigned int height, unsigned int width, unsigned int channels);
        Image(const std::string& path);
        Image(const Tensor<unsigned char>& tensor);
        Image(const Tensor<double>& tensor);
        ~Image();
    public:
        Tensor<double> as_tensor() const;
        unsigned int height() const;
        unsigned int width() const;
        unsigned int channels() const;
        Image as_grayscale() const;
        Image as_RGB() const;
        Image as_RGBA() const;
        Image resized(unsigned int height, unsigned int width) const;
        void save(const std::string& path) const;
    public:
        Image& operator=(const Image& other);
    };

    //Python interface
    extern "C" Image* new_Image_from_path(char* path);
    extern "C" Image* new_Image_from_data(unsigned char* data, unsigned int height, unsigned int width, unsigned int channels);
    extern "C" unsigned int Image_height(Image* image);
    extern "C" unsigned int Image_width(Image* image);
    extern "C" unsigned int Image_channels(Image* image);
    extern "C" Image* Image_as_grayscale(Image* image);
    extern "C" Image* Image_as_RGB(Image* image);
    extern "C" Image* Image_as_RGBA(Image* image);
    extern "C" Image* Image_resized(Image* image, unsigned int height, unsigned int width);
    extern "C" void Image_save(Image* image, char* path);
    extern "C" void Image_copy(Image* from, Image* to);
    extern "C" Tensor<double>* Image_as_Tensor(Image* image);
    extern "C" void del_Image(Image* image);
}