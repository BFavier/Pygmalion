#include <image/image.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <image/stb_image.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <image/stb_image_write.hpp>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <image/stb_image_resize.hpp>

using namespace pygmalion;

Image::Image()
{
}

Image::Image(const Image& other) : Tensor<unsigned char>(other)
{
}

Image::Image(unsigned int h, unsigned int w, unsigned int c) : Tensor<unsigned char>({h, w, c})
{
}

Image::Image(std::vector<unsigned char> d, unsigned int h, unsigned int w, unsigned int c) : Tensor<unsigned char>(d.data(), {h, w, c})
{
}

Image::Image(const std::string& path)
{
    int _w, _h, _c;
    unsigned char* pixels = stbi_load(path.c_str(), &_w, &_h, &_c, 0);
    if (pixels == nullptr)
    {
        throw std::runtime_error("Failed to load image from path '"+path+"': "+stbi_failure_reason());
    }
    unsigned int W = static_cast<unsigned int>(_w);
    unsigned int H = static_cast<unsigned int>(_h);
    unsigned int C = static_cast<unsigned int>(_c);
    Tensor<unsigned char> tensor(pixels, {H, W, C});
    Tensor<unsigned char>::operator=(tensor);
    stbi_image_free(pixels);
}

Image::Image(const Tensor<unsigned char>& tensor) : Tensor<unsigned char>(tensor)
{
}

Image::Image(const Tensor<double>& tensor)
{
    if (tensor.shape.size() != 3)
    {
        throw std::runtime_error(std::string(__func__)+": Constructor expected a tensor with 3 dimensions but got "+std::to_string(tensor.shape.size()));
    }
    unsigned int C = tensor.shape[0];
    if (C != 1 && C != 3 && C != 4)
    {
        throw std::runtime_error(std::string(__func__)+": Unexpected number of channels");
    }
    unsigned int H = tensor.shape[1];
    unsigned int W = tensor.shape[2];
    Image image(H, W, C);
    unsigned char* I = image.address();
    double* T = tensor.address();
    for (unsigned int i=0; i<C; i++)
    {
        for (unsigned int j=0; j<H; j++)
        {
            for (unsigned int k=0; k<W; k++)
            {
                double scalar = std::round(T[(i*H + j)*W + k]);
                scalar = std::max(0., std::min(scalar, 255.));
                I[(j*W + k)*C + i] = static_cast<unsigned char>(scalar);
            }
        }
    }
    Image::operator=(image);
}

Image::~Image()
{
}

unsigned int Image::height() const
{
    return shape[0];
}

unsigned int Image::width() const
{
    return shape[1];
}

unsigned int Image::channels() const
{
    return shape[2];
}

Tensor<double> Image::as_tensor() const
{
    unsigned int C = channels();
    unsigned int H = height();
    unsigned int W = width();
    Tensor<double> output({C, H, W});
    double* O = output.address();
    unsigned char* I = address();
    for (unsigned int i=0; i<C; i++)
    {
        for (unsigned int j=0; j<H; j++)
        {
            for (unsigned int k=0; k<W; k++)
            {
                unsigned int i_index = (j*W + k)*C + i;
                unsigned int o_index = (i*H + j)*W + k;
                O[o_index] = static_cast<double>(I[i_index]);
            }
        }
    }
    return output;
}

Image Image::as_grayscale() const
{
    unsigned int c = channels();
    unsigned int w = width();
    unsigned int h = height();
    // If this image is already a grayscale image
    if (c == 1)
    {
        return *this;
    }
    // Otherwise
    Image output(h, w, 1);
    unsigned char* O = output.address();
    unsigned char* I = address();
    for (unsigned int i=0; i<h; i++)
    {
        for (unsigned int j=0; j<w; j++)
        {
            unsigned int index = i*w + j;
            double pixel = 0;
            // Looping on [R, G, B] only even for RGBA images
            for (unsigned int k=0; k<3; k++)
            {
                pixel += I[index*c + k]/3.;
            }
            O[index] = static_cast<unsigned char>(std::round(pixel));
        }
    }
    return output;
}

Image Image::as_RGB() const
{
    unsigned int c = channels();
    // If this image is already an RGB image
    if (c == 3)
    {
        return *this;
    }
    // Otherwise
    unsigned int w = width();
    unsigned int h = height();
    Image output(h, w, 3);
    unsigned char* O = output.address();
    unsigned char* I = address();
    // If this image is an RGBA image
    if (c == 4)
    {
        for (unsigned int i=0; i<h; i++)
        {
            for (unsigned int j=0; j<w; j++)
            {
                unsigned int index = (i*w + j);
                for (unsigned int k=0; k<3; k++)
                {
                    O[index*3+k] = I[index*4+k];
                }
            }
        }
    }
    // If this image is a grayscale image
    else if (c == 1)
    {
        for (unsigned int i=0; i<h; i++)
        {
            for (unsigned int j=0; j<w; j++)
            {
                unsigned int index = (i*w + j);
                for (unsigned int k=0; k<3; k++)
                {
                    O[index*3+k] = I[index];
                }
            }
        }
    }
    return output;
}

Image Image::as_RGBA() const
{
    unsigned int c = channels();
    // If this image is already an RGBA image
    if (c == 4)
    {
        return *this;
    }
    // Otherwise
    unsigned int w = width();
    unsigned int h = height();
    Image output(h, w, 4);
    unsigned char* O = output.address();
    unsigned char* I = address();
    // If this image is an RGB image
    if (c == 3)
    {
        for (unsigned int i=0; i<h; i++)
        {
            for (unsigned int j=0; j<w; j++)
            {
                unsigned int index = (i*w + j);
                for (unsigned int k=0; k<3; k++)
                {
                    O[index*4+k] = I[index*3+k];
                }
                O[index*4+3] = 255;
            }
        }
    }
    // If this image is a grayscale image
    else if (c == 1)
    {
        for (unsigned int i=0; i<h; i++)
        {
            for (unsigned int j=0; j<w; j++)
            {
                unsigned int index = (i*w + j);
                for (unsigned int k=0; k<3; k++)
                {
                    O[index*4+k] = I[index];
                }
                O[index*4+3] = 255;
            }
        }
    }
    return output;
}

Image Image::resized(unsigned int h, unsigned int w) const
{
    unsigned int c = channels();
    Image output(h, w, c);
    stbir_resize_uint8(address(), width(), height(), 0,
                       output.address(), w, h, 0, c);
    return output;
}

void Image::save(const std::string& path) const
{
    // Getting the file extension
    std::string extension = path.substr(path.find_last_of("."));
    // Converting to lowercase
    for (unsigned int i=0; i<extension.size(); i++)
    {
        extension[i] = std::tolower(extension[i]);
    }
    // Saving the file
    int flag;
    if (extension == ".jpg" || extension == ".jpeg")
    {
        flag = stbi_write_jpg(path.c_str(),
                              static_cast<int>(width()),
                              static_cast<int>(height()),
                              static_cast<int>(channels()),
                              address(), 100);
    }
    else if (extension == ".png")
    {
        flag = stbi_write_png(path.c_str(),
                              static_cast<int>(width()),
                              static_cast<int>(height()),
                              static_cast<int>(channels()),
                              address(), 0);
    }
    else if (extension == ".bmp")
    {
        flag = stbi_write_bmp(path.c_str(),
                              static_cast<int>(width()),
                              static_cast<int>(height()),
                              static_cast<int>(channels()),
                              address());
    }
    else
    {
        throw std::runtime_error("Unsupported image format '"+extension+"'");
    }
    if (flag != 1)
    {
        throw std::runtime_error("Failed to save the output file '"+path+"': flag="+std::to_string(flag));
    }
}

Image& Image::operator=(const Image& other)
{
    Tensor<unsigned char>::operator=(other);
    return *this;
}


//Python interface
extern "C" Image* new_Image_from_path(char* path)
{
    return new Image(std::string(path));
}

extern "C" Image* new_Image_from_data(unsigned char* data, unsigned int height, unsigned int width, unsigned int channels)
{
    unsigned int L = height*width*channels;
    std::vector<unsigned char> vec;
    vec.reserve(L);
    for (unsigned int i=0; i<L; i++)
    {
        vec.push_back(data[i]);
    }
    return new Image(vec, height, width, channels);
}

extern "C" unsigned int Image_height(Image* image)
{
    return image->height();
}

extern "C" unsigned int Image_width(Image* image)
{
    return image->width();
}

extern "C" unsigned int Image_channels(Image* image)
{
    return image->channels();
}

extern "C" Image* Image_as_grayscale(Image* image)
{
    return new Image(image->as_grayscale());
}

extern "C" Image* Image_as_RGB(Image* image)
{
    return new Image(image->as_RGB());
}

extern "C" Image* Image_as_RGBA(Image* image)
{
    return new Image(image->as_RGBA());
}

extern "C" Image* Image_resized(Image* image, unsigned int height, unsigned int width)
{
    return new Image(image->resized(height, width));
}

extern "C" void Image_save(Image* image, char* path)
{
    image->save(std::string(path));
}

extern "C" void Image_copy(Image* from, Image* to)
{
    *from = *to;
}

extern "C" Tensor<double>* Image_as_Tensor(Image* image)
{
    return new Tensor<double>(image->as_tensor());
}

extern "C" void del_Image(Image* image)
{
    delete image;
}
