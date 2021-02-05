#include <neural_networks/layers/layers.hpp>
using namespace pygmalion;

const std::map<std::string, TensorOperation> machine_learning::functions = {{"identity", &identity},
                                                                            {"relu", &relu},
                                                                            {"tanh", &tanh}};

Tensor<double> machine_learning::normalize(const Tensor<double>& input, const Tensor<double>& mean, const Tensor<double>& standard_deviation)
{
    // Checking parameters
    if (input.shape.size() < 1 || mean.shape.size() != 1 || standard_deviation.shape.size() != 1)
    {
        throw std::runtime_error(std::string(__func__) + ": Parameters have an unexpected number of dimensions");
    }
    if (input.shape[0] != mean.shape[0] || mean.shape[0] != standard_deviation.shape[0])
    {
        throw std::runtime_error(std::string(__func__) + ": Shape of parameters doesn't match");
    }
    // Reading shape
    const unsigned int& C = input.shape[0];
    unsigned int L = input.size/C;
    // Allocating the output
    Tensor<double> output(input.shape);
    // Getting the raw addresses
    double* I = input.address();
    double* O = output.address();
    double* M = mean.address();
    double* S = standard_deviation.address();
    // Normalizing the tensor
    for (unsigned int c=0; c<C; c++)
    {
        for (unsigned int l=0; l<L; l++)
        {
            O[c*L + l] = (I[c*L + l] - M[c])/S[c];
        }
    }
    return output;
}

Tensor<double> machine_learning::pad(const Tensor<double>& input, double value, unsigned int left, unsigned int right, unsigned int top, unsigned int bottom)
{
    // Checking parameters
    if (input.shape.size() != 3)
    {
        throw std::runtime_error(std::string(__func__) + ": Input has an unexpected number of dimensions");
    }
    //Reading shape
    const unsigned int& C = input.shape[0];
    const unsigned int& H_in = input.shape[1];
    const unsigned int& W_in = input.shape[2];
    //Calculating output shape
    unsigned int H_out = H_in+top+bottom;
    unsigned int W_out = W_in+left+right;
    //if the shape is the same
    if (H_out == H_in && W_out == W_in)
    {
        return input;
    }
    //Allocating output memory
    Tensor<double> output({C, H_out, W_out});
    //Accessing the raw memory
    double* I = input.address();
    double* O = output.address();
    //Looping on channels
    for (unsigned int c=0; c<C; c++)
    {
        //Filling the lines at the top
        std::fill(&O[c*H_out*W_out], &O[(c*H_out+top)*W_out], value);
        //Looping on lines
        for (unsigned int y=0; y<H_in; y++)
        {
            unsigned int offset = (c*H_out + y+top)*W_out;
            //Filling the columns on the left
            std::fill(&O[offset], &O[offset + left], value);
            //Copying the line
            std::copy(&I[(c*H_in + y)*W_in], &I[(c*H_in + y)*W_in + W_in], &O[offset + left]);
            //Filling the columns on the right
            std::fill(&O[offset + left + W_in], &O[offset + W_out], value);
        }
        //Filling lines at the bottom
        std::fill(&O[(c*H_out+top+H_in)*W_out], &O[(c+1)*H_out*W_out], value);
    }
    //return the result
    return output;
}

Tensor<double> machine_learning::unroll(const Tensor<double>& input, unsigned int H_kernel, unsigned int W_kernel, unsigned int S_h, unsigned int S_w)
{
    // Checking parameters
    if (input.shape.size() != 3)
    {
        throw std::runtime_error(std::string(__func__) + ": Input has an unexpected number of dimensions");
    }
    //Reading shape
    const unsigned int& C = input.shape[0];
    const unsigned int& H_in = input.shape[1];
    const unsigned int& W_in = input.shape[2];
    //Calculate the output shape
    unsigned int H_out = (H_in - H_kernel)/S_h + 1;
    unsigned int W_out = (W_in - W_kernel)/S_w + 1;
    //Allocating the unrolled tensor
    Tensor<double> unrolled({H_out, W_out, C, H_kernel, W_kernel});
    //Getting the raw addresses
    double* I = input.address();
    double* U = unrolled.address();
    //Looping on the input
    for (unsigned int c=0; c<C; c++)
    {
        for (unsigned int y_out=0; y_out<H_out; y_out++)
        {
            unsigned int y = y_out * S_h;
            for (unsigned int x_out=0; x_out<W_out; x_out++)
            {
                unsigned int x = x_out*S_w;
                //copying the content of the window
                for (unsigned int yy=0; yy<H_kernel; yy++)
                {
                    unsigned int o_index = (((y_out*W_out + x_out)*C + c)*H_kernel + yy)*W_kernel;
                    unsigned int i_index = (c*H_in + y + yy)*W_in + x;
                    std::copy(&I[i_index], &I[i_index+W_kernel], &U[o_index]);
                }
            }
        }
    }
    return unrolled;
}

Tensor<double> machine_learning::convolve(const Tensor<double>& input, const Tensor<double>& kernel, const Tensor<double>& bias, unsigned int S_h, unsigned int S_w)
{
    // Checking parameters
    if (input.shape.size() != 3 || kernel.shape.size() != 4 || bias.shape.size() != 1)
    {
        throw std::runtime_error(std::string(__func__) + ": Parameters have an unexpected number of dimensions");
    }
    if (input.shape[0] != kernel.shape[1] || kernel.shape[0] != bias.shape[0])
    {
        throw std::runtime_error(std::string(__func__) + ": Shape of parameters doesn't match");
    }
    if (input.shape[1] < kernel.shape[2] || input.shape[2] < kernel.shape[3])
    {
        throw std::runtime_error(std::string(__func__) + ": The convolved kernel can not have an height/width superior to the input's height/width");
    }
    //Reading shape
    const unsigned int& C_in = input.shape[0];
    const unsigned int& H_in = input.shape[1];
    const unsigned int& W_in = input.shape[2];
    const unsigned int& C_out = kernel.shape[0];
    const unsigned int& H_kernel = kernel.shape[2];
    const unsigned int& W_kernel = kernel.shape[3];
    unsigned int L = C_in*H_kernel*W_kernel;
    //Calculate the output shape
    unsigned int H_out = (H_in - H_kernel)/S_h + 1;
    unsigned int W_out = (W_in - W_kernel)/S_w + 1;
    //Unrolling the input
    Tensor<double> unrolled(unroll(input, H_kernel, W_kernel, S_h, S_w));
    //Allocating the result
    Tensor<double> output({C_out, H_out, W_out});
    //Getting the unrolled/kernel/output addresses
    double* U = unrolled.address();
    double* K = kernel.address();
    double* B = bias.address();
    double* O = output.address();
    //Calculating the output
    for(unsigned int c=0; c<C_out; c++)
    {
        for (unsigned int y=0; y<H_out; y++)
        {
            for (unsigned int x=0; x<W_out; x++)
            {
                unsigned int o_index = (c*H_out + y)*W_out + x;
                unsigned int u_index = (y*W_out + x)*L;
                O[o_index] = B[c];
                for (unsigned int k=0; k<L; k++)
                {
                    O[o_index] += K[k+c*L]*U[u_index+k];
                }
            }
        }
    }
    return output;
}

Tensor<double> machine_learning::max_pool(const Tensor<double>& input, unsigned int height, unsigned int width)
{
    // Checking parameters
    if (input.shape.size() != 3)
    {
        throw std::runtime_error(std::string(__func__) + ": Input has an unexpected number of dimensions");
    }
    if (input.shape[1] < height || input.shape[2] < width)
    {
        throw std::runtime_error(std::string(__func__) + ": The max pooling window's height/width can not be superior to the input's height/width");
    }
    // If the max pooling has no effect
    if (height == 1 && width == 1)
    {
        return input;
    }
    //Reading shape
    const unsigned int& C = input.shape[0];
    const unsigned int& H_in = input.shape[1];
    const unsigned int& W_in = input.shape[2];
    //Calculate the output shape
    unsigned int H_out = H_in/height;
    unsigned int W_out = W_in/width;
    //Allocating the result
    Tensor<double> output({C, H_out, W_out});
    //Getting the raw addresses
    double* O = output.address();
    double* I = input.address();
    //Filling the result
    for (unsigned int c=0; c<C; c++)
    {
        for (unsigned int y=0; y<H_out; y++)
        {
            for (unsigned x=0; x<W_out; x++)
            {
                unsigned int o_index = (c*H_out + y)*W_out + x;
                O[o_index] = -std::numeric_limits<double>::infinity();
                //Looping on the original image
                for (unsigned int yy=0; yy<height; yy++)
                {
                    for (unsigned int xx=0; xx<width; xx++)
                    {
                        unsigned int i_index = (c*H_in + y*height + yy)*W_in + x*width + xx;
                        if (I[i_index] > O[o_index])
                        {
                            O[o_index] = I[i_index];
                        }
                    }
                }
            }
        }
    }
    return output;
}

Tensor<double> machine_learning::batch_normalize(const Tensor<double>& input, const Tensor<double>& mean, const Tensor<double>& variance, const Tensor<double>& weight, const Tensor<double>& bias)
{
    // Checking parameters
    if (input.shape.size() != 3 || mean.shape.size() != 1 || variance.shape.size() != 1 || weight.shape.size() != 1 || bias.shape.size() != 1)
    {
        throw std::runtime_error(std::string(__func__) + ": Parameters have an unexpected number of dimensions");
    }
    if (input.shape[0] != mean.shape[0] || input.shape[0] != variance.shape[0] || input.shape[0] != weight.shape[0] || input.shape[0] != bias.shape[0])
    {
        throw std::runtime_error(std::string(__func__) + ": Shape of parameters doesn't match");
    }
    //Reading shape
    const unsigned int& C = input.shape[0];
    const unsigned int& H = input.shape[1];
    const unsigned int& W = input.shape[2];
    //Allocating
    Tensor<double> output({C, H, W});
    Tensor<double> std(variance.shape);
    //Accessing raw memory
    double* I = input.address();
    double* O = output.address();
    double* V = variance.address();
    double* S = std.address();
    double* M = mean.address();
    double* A = weight.address();
    double* B = bias.address();
    //Calculating std
    for (unsigned int i=0; i<std.size; i++)
    {
        S[i] = std::sqrt(V[i]+1.0E-5);
    }
    //Filling results
    for (unsigned int c=0; c<C; c++)
    {
        for (unsigned int y=0; y<H; y++)
        {
            for (unsigned int x=0; x<W; x++)
            {
                unsigned int index = ((c*H)+y)*W+x;
                O[index] = (I[index] - M[c])/S[c] * A[c] + B[c];
            }
        }
    }
    //Returning results
    return output;
}

Tensor<double> machine_learning::linear(const Tensor<double>& input, const Tensor<double>& weights, const Tensor<double>& bias)
{
    // Checking parameters
    if (input.shape.size() < 1 || weights.shape.size() != 2 || bias.shape.size() != 1)
    {
        throw std::runtime_error(std::string(__func__) + ": Parameters have an unexpected number of dimensions");
    }
    if (input.shape[0] != weights.shape[1] || weights.shape[0] != bias.shape[0])
    {
        throw std::runtime_error(std::string(__func__) + ": Shape of parameters doesn't match");
    }
    //Reading shape
    const unsigned int& C_in = input.shape[0];
    const unsigned int& C_out = bias.shape[0];
    //Calculating number of linear operations to perform
    unsigned int L = input.size / C_in;
    //Allocating
    std::vector<unsigned int> shape_out = input.shape;
    shape_out[0] = C_out;
    Tensor<double> output(shape_out);
    //Accessing raw memory
    double* I = input.address();
    double* O = output.address();
    double* W = weights.address();
    double* B = bias.address();
    //Filling result
    for (unsigned int i=0; i<C_out; i++)
    {
        for (unsigned int l=0; l<L; l++)
        {
            O[i*L + l] = B[i];
            for (unsigned int j=0; j<C_in; j++)
            {
                O[i*L + l] += I[j*L + l]*W[i*C_in + j];
            }
        }
    }
    //return the result
    return output;
}

Tensor<double> machine_learning::resample(const Tensor<double>& input, unsigned int new_height, unsigned int new_width)
{
    // Checking parameters
    if (input.shape.size() != 3)
    {
        throw std::runtime_error(std::string(__func__) + ": Input has an unexpected number of dimensions");
    }
    // Reading shape
    const unsigned int& channels = input.shape[0];
    const unsigned int& height = input.shape[1];
    const unsigned int& width = input.shape[2];
    Tensor<double> output({channels, new_height, new_width});
    //Accessing raw memory
    double* I = input.address();
    double* O = output.address();
    //Filling result
    for (unsigned int c=0; c<channels; c++)
    {
        for (unsigned int y_new=0; y_new<new_height; y_new++)
        {
            for (unsigned int x_new=0; x_new<new_width; x_new++)
            {
                // Calculate relative x, y coordinates in the image
                double rx = (static_cast<double>(x_new) + 0.5)/new_width;
                double ry = (static_cast<double>(y_new) + 0.5)/new_height;
                // Calculate the x, y coordinates in the input image
                double x = rx*width - 0.5;
                double y = ry*height - 0.5;
                // Find the coordinates of the 4 closest pixels in the original image
                unsigned int x_left = std::min(static_cast<unsigned int>(std::max(0., std::floor(x))), width-1);
                unsigned int x_right = std::min(static_cast<unsigned int>(std::max(0., std::ceil(x))), width-1);
                unsigned int y_top = std::min(static_cast<unsigned int>(std::max(0., std::floor(y))), height-1);
                unsigned int y_bot = std::min(static_cast<unsigned int>(std::max(0., std::ceil(y))), height-1);
                // Calculate the value of the corresponding pixel in input image
                double z_top_left = I[(c*height + y_top)*width + x_left];
                double z_top_right = I[(c*height + y_top)*width + x_right];
                double z_bot_left = I[(c*height + y_bot)*width + x_left];
                double z_bot_right = I[(c*height + y_bot)*width + x_right];
                // Claculate the offset in-between the 4 pixels
                double tx = x - static_cast<double>(x_left);
                double ty = y - static_cast<double>(y_top);
                // Interpolate
                O[(c*new_height + y_new)*new_width + x_new] = tx*ty*z_bot_right +
                                                              tx*(1 - ty)*z_top_right +
                                                              (1 - tx)*ty*z_bot_left +
                                                              (1 - tx)*(1 - ty)*z_top_left;
            }
        }
    }
    return output;
}

Tensor<double> machine_learning::identity(const Tensor<double>& input)
{
    return input;
}

Tensor<double> machine_learning::relu(const Tensor<double>& input)
{
    Tensor<double> output(input.shape);
    double* I = input.address();
    double* O = output.address();
    for (unsigned int i=0; i<input.size; i++)
    {
        O[i] = std::max(0., I[i]);
    }
    return output;
}

Tensor<double> machine_learning::tanh(const Tensor<double>& input)
{
    Tensor<double> output(input.shape);
    double* I = input.address();
    double* O = output.address();
    for (unsigned int i=0; i<input.size; i++)
    {
        O[i] = std::tanh(I[i]);
    }
    return output;
}

//python interface
extern "C" Tensor<double>* normalize(Tensor<double>* input, Tensor<double>* mean, Tensor<double>* standard_deviation)
{
    return new Tensor<double>(normalize(*input, *mean, *standard_deviation));
}

extern "C" Tensor<double>* pad(Tensor<double>* input, double value, unsigned int left, unsigned int right, unsigned int top, unsigned int bottom)
{
    return new Tensor<double>(pad(*input, value, left, right, top, bottom));
}

extern "C" Tensor<double>* convolve(Tensor<double>* input, Tensor<double>* kernel, Tensor<double>* bias, unsigned int stride_height, unsigned int stride_width)
{
    return new Tensor<double>(convolve(*input, *kernel, *bias, stride_height, stride_width));
}

extern "C" Tensor<double>* max_pool(Tensor<double>* input, unsigned int height, unsigned int width)
{
    return new Tensor<double>(max_pool(*input, height, width));
}

extern "C" Tensor<double>* batch_normalize(Tensor<double>* input, Tensor<double>* mean, Tensor<double>* variance, Tensor<double>* weight, Tensor<double>* bias)
{
    return new Tensor<double>(batch_normalize(*input, *mean, *variance, *weight, *bias));
}

extern "C" Tensor<double>* linear(Tensor<double>* input, Tensor<double>* weights, Tensor<double>* bias)
{
    return new Tensor<double>(linear(*input, *weights, *bias));
}

extern "C" Tensor<double>* resample(Tensor<double>* input, unsigned int new_height, unsigned int new_width)
{
    return new Tensor<double>(resample(*input, new_height, new_width));
}

extern "C" Tensor<double>* Tensor_relu(Tensor<double>* input)
{
    return new Tensor<double>(relu(*input));
}

extern "C" Tensor<double>* Tensor_tanh(Tensor<double>* input)
{
    return new Tensor<double>(tanh(*input));
}
