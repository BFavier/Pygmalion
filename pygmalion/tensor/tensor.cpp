#include <tensor/tensor.hpp>
using namespace pygmalion;
using namespace nlohmann;

template <typename T>
Tensor<T>::Tensor()
{
    _allocate();
}

template <typename T>
Tensor<T>::Tensor(const Tensor<T>& other) : shape{other.shape}, offset{other.offset}
{
    _set_size();
    data = other.data;
}

template <typename T>
Tensor<T>::Tensor(T value)
{
    _allocate();
    data[0] = value;
}

template <typename T>
Tensor<T>::Tensor(const std::initializer_list<T>& content, const std::initializer_list<unsigned int>& shape_) : shape{shape_}
{
    _allocate();
    unsigned int i=0;
    for (T value : content)
    {
        data[i] = value;
        i++;
    }
}

template <typename T>
Tensor<T>::Tensor(const std::initializer_list<unsigned int>& shape_) : shape{shape_}
{
    _allocate();
}

template <typename T>
Tensor<T>::Tensor(const std::vector<unsigned int>& shape_) : shape{shape_}
{
    _allocate();
}

template <typename T>
Tensor<T>::Tensor(const std::vector<unsigned int>& shape_, T value) : shape{shape_}
{
    _allocate();
    for (unsigned int i=0; i<size; i++)
    {
        data[i] = value;
    }
}

template <typename T>
Tensor<T>::Tensor(const std::shared_ptr<T[]>& data_, const std::vector<unsigned int>& shape_, unsigned int offset_) : shape{shape_}, offset{offset_}
{
    _set_size();
    data = data_;
}


template <typename T>
Tensor<T>::Tensor(T* array, const std::vector<unsigned int>& shape_) : shape{shape_}
{
    _allocate();
    for (unsigned int i=0; i<size; i++)
    {
        data[i] = array[i];
    }
}

template <typename T>
Tensor<T>::Tensor(const json& json_object)
{
    const json* pointer = &json_object;
    // Look up the shape of the nested arrays
    while (pointer->is_array())
    {
        shape.push_back(pointer->size());
        pointer = &pointer->at(0);
    }
    _allocate();
    // Fill the tensor
    if (shape.empty())
    {
        data[0] = *(pointer);
        return;
    }
    std::vector<unsigned int> index(shape.size(), 0);
    do
    {
        pointer = &json_object;
        for (unsigned int i : index)
        {
            pointer = &pointer->at(i);
        }
        (*this)[index] = *pointer;
    }
    while (increment(index));
}


template <typename T>
Tensor<T>::~Tensor()
{
}

template <typename T>
Tensor<T> Tensor<T>::copy() const
{
    Tensor<T> copied(shape);
    std::copy(address(), &address()[size], copied.address());
    return copied;
}

template <typename T>
void Tensor<T>::copy(const Tensor<T>& other)
{
    if (shape != other.shape)
    {
        _throw_bad_shape(__func__);
    }
    T* A = address();
    T* B = other.address();
    for (unsigned int i=0; i<size; i++)
    {
        A[i] = B[i];
    }
}

template <typename T>
Tensor<T> Tensor<T>::reshape(const std::vector<unsigned int>& new_shape) const
{
    Tensor<T> reshaped(data, new_shape, offset);
    if (reshaped.size != size)
    {
        throw std::runtime_error("Tried to reshape a tensor to a new shape that doesn't have the same total number of elements");
    }
    return reshaped;
}

template <typename T>
Tensor<T> Tensor<T>::flatten() const
{
    std::vector<unsigned int> new_shape({size});
    return reshape(new_shape);
}

template <typename T>
T* Tensor<T>::address() const
{
    return data.get() + offset;
}

template <typename T>
bool Tensor<T>::increment(std::vector<unsigned int>& coords) const
{
    if (coords.size() != shape.size())
    {
        _throw_bad_indexing(__func__);
    }
    unsigned int D = shape.size();
    for (unsigned int i=D-1; i<D; i--)
    {
        coords[i]++;
        if (coords[i] == shape[i])
        {
            coords[i] = 0;
        }
        else if (coords[i] < shape[i])
        {
            return true;
        }
        else
        {
            _throw_bad_indexing(__func__);
        }
    }
    return false;
}

template <typename T>
Tensor<unsigned int> Tensor<T>::index_max(unsigned int axis) const
{
    // Calculating the output shape, and temporary shapes
    std::vector<unsigned int> out_shape;
    unsigned int L_before = 1;
    unsigned int L_axis = shape[axis];
    unsigned int L_after = 1;
    for (unsigned int i=0; i<axis; i++)
    {
        L_before *= shape[i];
        out_shape.push_back(shape[i]);
    }
    for (unsigned int i=axis+1; i<shape.size(); i++)
    {
        L_after *= shape[i];
        out_shape.push_back(shape[i]);
    }
    Tensor<T> out_max({L_before, L_after}, std::numeric_limits<T>::lowest());
    Tensor<unsigned int> out_index(out_shape, 0);
    // Access the raw memory
    T* I = address();
    T* O_max = out_max.address();
    unsigned int* O_index = out_index.address();
    // Fill the output tensor
    for (unsigned int before=0; before<L_before; before++)
    {
        for (unsigned int i=0; i<L_axis; i++)
        {
            for (unsigned int after=0; after<L_after; after++)
            {
                unsigned int index_I = (before*L_axis + i)*L_after + after;
                unsigned int index_O = before*L_after + after;
                if (I[index_I] > O_max[index_O])
                {
                    O_max[index_O] = I[index_I];
                    O_index[index_O] = i;
                }
            }
        }
    }
    // Reshape the output tensor
    return out_index;
}

template <typename T>
Tensor<T> Tensor<T>::at_index(const Tensor<unsigned int>& coords) const
{
    // flattening the coordinates into a 1D tensor
    const Tensor<unsigned int> flat_coords = coords.flatten();
    // Defining output shape
    unsigned int L = 1;
    std::vector<unsigned int> out_shape(coords.shape);
    for (unsigned int i=1; i<shape.size(); i++)
    {
        unsigned int l = shape[i];
        out_shape.push_back(l);
        L *= l;
    }
    // reshaping the indexed tensor
    const Tensor<T> flat_tensor = this->reshape({shape[0], L});
    // Allocating the tensor
    Tensor<T> output({flat_coords.size, L});
    // Filling the tensor
    for (unsigned int i=0; i<flat_coords.size; i++)
    {
        unsigned int index = flat_coords[{i}];
        for (unsigned int j=0; j<L; j++)
        {
            output[{i, j}] = flat_tensor[{index, j}];
        }
    }
    // Reshape the output tensor
    return output.reshape(out_shape);
}

template <typename T>
Tensor<T> Tensor<T>::concatenate(const std::vector<Tensor<T>>& tensors, unsigned int axis)
{
    // Checking parameters
    if (tensors.empty())
    {
        throw std::runtime_error(std::string(__func__) + ": Empty vector of tensors passed as argument.");
    }
    for (unsigned int i=1; i<tensors.size(); i++)  // Looping on tensors past the first one
    {
        // All tensors must have the same dimension
        if (tensors[i].shape.size() != tensors[0].shape.size())
        {
            throw std::runtime_error(std::string(__func__) + ": All tensors must have the same dimension");
        }
        // For each dimension
        for (unsigned int j=0; j<tensors[0].shape.size(); j++)
        {
            // excepted for the concatenation axis
            if (j == axis)
            {
                continue;
            }
            // the two dimension mst have the same length
            if (tensors[i].shape[j] != tensors[0].shape[j])
            {
                throw std::runtime_error(std::string(__func__) + ": Shape of tensors missmatch");
            }
        }
    }
    // Calculating the intermediate shape
    unsigned int L_before = 1;
    unsigned int L_axis = 0;
    unsigned int L_after = 1;
    for (unsigned int i=0; i<axis; i++)
    {
        L_before *= tensors[0].shape[i];
    }
    for (const Tensor<T>& tensor : tensors)
    {
        L_axis += tensor.shape[axis];
    }
    for (unsigned int i=axis+1; i<tensors[0].shape.size(); i++)
    {
        L_after *= tensors[0].shape[i];
    }
    std::vector<unsigned int> new_shape = {L_before, L_axis, L_after};
    // Calculating the output shape
    std::vector<unsigned int> out_shape(tensors[0].shape);
    out_shape[axis] = L_axis;
    // Allocate output
    Tensor<T> output(out_shape);
    T* O = output.address();
    // Fill the output
    for (unsigned int before=0; before<L_before; before++)
    {
        unsigned int offset = 0;
        for (const Tensor<T>& tensor : tensors)
        {
            unsigned int l_axis = tensor.shape[axis];
            T* I = tensor.address();
            for (unsigned int i=0; i<l_axis; i++)
            {
                for (unsigned int after=0; after<L_after; after++)
                {
                    O[(before*L_axis + offset+i)*L_after + after] = I[(before*l_axis + i)*L_after + after];
                }
            }
            offset += l_axis;
        }
    }
    return output;
}

template <typename T>
Tensor<T> Tensor<T>::operator[](unsigned int i) const
{
    unsigned int mult = 1;
    std::vector<unsigned int> new_shape;
    new_shape.reserve(shape.size());
    for (unsigned int i=1; i<shape.size(); i++)
    {
        new_shape.push_back(shape[i]);
        mult *= shape[i];
    }
    return Tensor<T>(data, new_shape, offset+mult*i);
}

template <typename T>
T& Tensor<T>::operator[](const std::vector<unsigned int>& coords)
{
    unsigned int index = Tensor<T>::_coords_to_index(coords);
    return data[index+offset];
}

template <typename T>
T Tensor<T>::operator[](const std::vector<unsigned int>& coords) const
{
    unsigned int index = Tensor<T>::_coords_to_index(coords);
    return data[index+offset];
}

template <typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const
{
    return _binary_operator(*this, other, _add);
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T>& other) const
{
    return _binary_operator(*this, other, _sub);
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const
{
    return _binary_operator(*this, other, _mul);
}

template <typename T>
Tensor<T> Tensor<T>::operator/(const Tensor<T>& other) const
{
    return _binary_operator(*this, other, _div);
}

template <typename T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T>& other)
{
    _inplace_operator(other, _add);
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T>& other)
{
    _inplace_operator(other, _sub);
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator*=(const Tensor<T>& other)
{
    _inplace_operator(other, _mul);
    return *this;
}

template <typename T>
Tensor<T>& Tensor<T>::operator/=(const Tensor<T>& other)
{
    _inplace_operator(other, _div);
    return *this;
}

template <typename T>
bool Tensor<T>::operator==(const Tensor<T>& other) const
{
    if (shape != other.shape)
    {
        return false;
    }
    T* A = address();
    T* B = other.address();
    for (unsigned int i=0; i<size; i++)
    {
        if (A[i] != B[i])
        {
            return false;
        }
    }
    return true;
}

template <typename T>
Tensor<T>& Tensor<T>::operator=(const Tensor<T>& other)
{
    data = other.data;
    shape = other.shape;
    size = other.size;
    offset = other.offset;
    return *this;
}

template <typename T>
Tensor<T>::operator T() const
{
    if (!shape.empty())
    {
        throw std::runtime_error(std::string(__func__)+": Only a 0 dimensional tensor can be converted to base type.");
    }
    return *address();
}

namespace pygmalion
{
    template <typename T>
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& other)
    {
        if (other.shape.size() == 0)
        {
            os << other.data[0+other.offset];
            return os;
        }
        for (unsigned int i=0; i<other.size; i++)
        {
            unsigned int n;
            unsigned int mult;
            //Count the number of brackets to open
            mult = 1;
            unsigned int n_brackets = 0;
            for (n=0; n<other.shape.size(); n++)
            {
                unsigned int k = other.shape.size() - n - 1;
                mult *= other.shape[k];
                if (i % mult == 0)
                {
                    n_brackets ++;
                }
                else
                {
                    break;
                }

            }
            if (n_brackets > 0)
            {
                for (unsigned int j=0; j<other.shape.size()-n_brackets; j++)
                {
                    os << " ";
                }
                for (unsigned int j=0; j<n_brackets; j++)
                {
                    os << "[";
                }
            }

            //display the number
            os << other.data[i+other.offset];
            //display the separator
            if ((i+1) % other.shape[other.shape.size() - 1] != 0)
            {
                os << ", ";
            }
            //Count the number of brackets to close
            mult = 1;
            for (n=0; n<other.shape.size(); n++)
            {
                unsigned int k = other.shape.size() - n - 1;
                mult *= other.shape[k];
                if ((i+1) % mult == 0)
                {
                    os << "]";
                }
                else
                {
                    break;
                }

            }
            //Line skips
            if (n >= 1)
            {
                os << ",\n";
            }
        }
        return os;
    }
}

template <typename T>
void Tensor<T>::_allocate()
{
    _set_size();
    data.reset(new T[size]);
}

template <typename T>
void Tensor<T>::_set_size()
{
    size = 1;
    for (unsigned int len : shape)
    {
        size *= len;
    }
}

template <typename T>
void Tensor<T>::_throw_bad_shape(const std::string& func_name)
{
    throw std::range_error(func_name+": shapes of the two tensors don't match");
}

template <typename T>
void Tensor<T>::_throw_bad_indexing(const std::string& func_name)
{
    throw std::range_error(func_name+": number of coordinates doesn't match tensor's shape");
}

template <typename T>
unsigned int Tensor<T>::_coords_to_index(const std::vector<unsigned int>& coords) const
{
    unsigned int L = shape.size();
    unsigned int index = 0;
    unsigned int mult = 1;
    for (unsigned int i=L-1; i<L; i--)
    {
        index += coords[i]*mult;
        mult *= shape[i];
    }
    return index;
}

template <typename T>
Tensor<T> Tensor<T>::_expand_to_dim(unsigned int dim) const
{
    if (shape.size() > dim)
    {
        throw std::runtime_error(std::string(__func__)+": tensor's dimension is higher than 'dim'");
    }
    std::vector<unsigned int> new_shape(dim-shape.size(), 1);
    for (unsigned int l : shape)
    {
        new_shape.push_back(l);
    }
    return this->reshape(new_shape);
}

template <typename T>
std::vector<unsigned int> Tensor<T>::_clamp_coords(const std::vector<unsigned int>& coords) const
{
    if (coords.size() > shape.size())
    {
        _throw_bad_indexing(__func__);
    }
    std::vector<unsigned int> new_coords(coords.size());
    for (unsigned int i=0; i<shape.size(); i++)
    {
        new_coords[i] = std::min(coords[i], shape[i]-1);
    }
    return new_coords;
}

template <typename T>
Tensor<T> Tensor<T>::_binary_operator(const Tensor<T>& left, const Tensor<T>& right, T (*func)(const T& left, const T& right))
{
    // If the two tensors have the same shape
    if (left.shape == right.shape)
    {
        Tensor<T> output(left.shape);
        T* flat_left = left.address();
        T* flat_right = right.address();
        T* flat_out = output.address();
        for (unsigned int i=0; i<left.size; i++)
        {
            flat_out[i] = func(flat_left[i], flat_right[i]);
        }
        return output;
    }
    // Otherwise some fancy indexing is necessary
    else
    {
        // Reshape the two tensors to the same dimension
        unsigned int D = std::max(left.shape.size(), right.shape.size());
        Tensor<T> expanded_left = left._expand_to_dim(D);
        Tensor<T> expanded_right = right._expand_to_dim(D);
        // Create some alias
        const std::vector<unsigned int>& shape_left = expanded_left.shape;
        const std::vector<unsigned int>& shape_right = expanded_right.shape;
        // Check that the two tensors have a compatible shape
        for (unsigned int i=0; i<D; i++)
        {
            if ((shape_left[i] != shape_right[i]) && (shape_left[i] != 1) && (shape_right[i] != 1))
            {
                _throw_bad_shape(__func__);
            }
        }
        // Allocate the output tensor
        std::vector<unsigned int> out_shape(D);
        for (unsigned int i=0; i<D; i++)
        {
            out_shape[i] = std::max(shape_left[i], shape_right[i]);
        }
        Tensor<T> output(out_shape);
        // Loop over the output tensor
        std::vector<unsigned int> index(D, 0);
        do
        {
            std::vector<unsigned int> index_left = expanded_left._clamp_coords(index);
            std::vector<unsigned int> index_right = expanded_right._clamp_coords(index);
            output[index] = func(expanded_left[index_left], expanded_right[index_right]);
        } while (output.increment(index));
        return output;
    }
}

template <typename T>
void Tensor<T>::_inplace_operator(const Tensor<T>& right, T (*func)(const T& left, const T& right))
{
    // If the two tensors have the same shape
    if (this->shape == right.shape)
    {
        T* flat_left = this->address();
        T* flat_right = right.address();
        for (unsigned int i=0; i<this->size; i++)
        {
            flat_left[i] = func(flat_left[i], flat_right[i]);
        }
    }
    // Otherwise some fancy indexing is necessary
    else
    {
        // Reshape the right tensor to the same dimension
        Tensor<T> expanded_right = right._expand_to_dim(this->shape.size());
        // Check that the two vectors have a compatible shape
        for (unsigned int i=0; i<this->shape.size(); i++)
        {
            if ((this->shape[i] != expanded_right.shape[i]) && (expanded_right.shape[i] != 1))
            {
                _throw_bad_shape(__func__);
            }
        }
        // Loop over the left tensor
        std::vector<unsigned int> index(this->shape.size(), 0);
        do
        {
            std::vector<unsigned int> index_right = expanded_right._clamp_coords(index);
            (*this)[index] = func((*this)[index], expanded_right[index_right]);
        } while (this->increment(index));
    }
}

template <typename T>
T Tensor<T>::_add(const T& left, const T& right)
{
    return left+right;
}

template <typename T>
T Tensor<T>::_sub(const T& left, const T& right)
{
    return left-right;
}

template <typename T>
T Tensor<T>::_mul(const T& left, const T& right)
{
    return left*right;
}

template <typename T>
T Tensor<T>::_div(const T& left, const T& right)
{
    return left/right;
}

// Python interface
extern "C" Tensor<double>* new_Tensor(double* array, unsigned int* shape, unsigned int n_dims)
{
    std::vector<unsigned int> tshape;
    for (unsigned int i=0; i<n_dims; i++)
    {
        tshape.push_back(shape[i]);
    }
    return new Tensor<double>(array, tshape);
}

extern "C" Tensor<double>* Tensor_copy(Tensor<double>* tensor)
{
    return new Tensor<double>(tensor->copy());
}

extern "C" Tensor<double>* Tensor_reshape(Tensor<double>* tensor, unsigned int* new_shape, unsigned int n_dims)
{
    std::vector<unsigned int> shape;
    shape.reserve(n_dims);
    for (unsigned int i=0; i<n_dims; i++)
    {
        shape.push_back(new_shape[i]);
    }
    return new Tensor<double>(tensor->reshape(shape));
}

extern "C" unsigned int Tensor_ndims(Tensor<double>* tensor)
{
    return tensor->shape.size();
}

extern "C" unsigned int* Tensor_shape(Tensor<double>* tensor)
{
    return &tensor->shape[0];
}

extern "C" double* Tensor_data(Tensor<double>* tensor)
{
    return tensor->address();
}

extern "C" void Tensor_set(Tensor<double>* tensor, unsigned int* where, double val)
{
    std::vector<unsigned int> where_(where, where+tensor->shape.size());
    (*tensor)[where_] = val;
}

extern "C" double Tensor_get(Tensor<double>* tensor, unsigned int* where)
{
    std::vector<unsigned int> where_(where, where+tensor->shape.size());
    return (*tensor)[where_];
}

extern "C" Tensor<double>* Tensor_subTensor(Tensor<double>* tensor, unsigned int index)
{
    return new Tensor<double>((*tensor)[index]);
}

extern "C" void del_Tensor(Tensor<double>* tensor)
{
    delete tensor;
}

extern "C" Tensor<double>* Tensor_concatenate(Tensor<double>** tensors, unsigned int n_tensors, unsigned int axis)
{
    std::vector<Tensor<double>> _tensors;
    for (unsigned int i=0; i<n_tensors; i++)
    {
        _tensors.push_back(*tensors[i]);
    }
    return new Tensor<double>(Tensor<double>::concatenate(_tensors, axis));
}

// Templates instanciation
template class Tensor<double>;
template std::ostream& machine_learning::operator<<(std::ostream& os, const Tensor<double>& other);
template class Tensor<unsigned int>;
template std::ostream& machine_learning::operator<<(std::ostream& os, const Tensor<unsigned int>& other);
template class Tensor<unsigned char>;
template std::ostream& machine_learning::operator<<(std::ostream& os, const Tensor<unsigned char>& other);
