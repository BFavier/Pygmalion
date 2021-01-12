#pragma once
#include <json/json.hpp>
#include <vector>
#include <list>
#include <iostream>
#include <initializer_list>
#include <memory>  // for std::shared_ptr

namespace pygmalion
{
    template <typename T>
    class Tensor
    {
    public:
        Tensor(); ///< Default constructor
        Tensor(const Tensor<T>& other); ///< Copy constructor (the underlying data is shared)
        Tensor(T value); ///< Constructs a scalar tensor
        Tensor(const std::initializer_list<unsigned int>& shape); ///< Initialize a tensor of given shape
        Tensor(const std::initializer_list<T>& content, const std::initializer_list<unsigned int>& shape); ///< Constructs a tensor from an initializer list and a shape
        Tensor(const std::vector<unsigned int>& shape); ///< Initialize a tensor of given shape
        Tensor(const std::vector<unsigned int>& shape, T value); ///< Initialize a tensor of given shape, and given initial value
        Tensor(const std::shared_ptr<T[]>& data, const std::vector<unsigned int>& shape, unsigned int offset); ///< Constructor to share memory of another tensor, but with a different shape
        Tensor(T* array, const std::vector<unsigned int>& shape); ///< Initialize a tensor by copying a C array (for python to C++ transfer)
        Tensor(const nlohmann::json& json_object); ///< To parse a tensor from a nlohmann::json object
        ~Tensor(); ///< Destructor
    public:
        std::vector<unsigned int> shape = {}; ///< The size of each dimension of the tensor
        unsigned int offset = 0; ///< The offset to apply to the allocated memory array
        unsigned int size = 1;  ///< The number of scalars that can be accessed in the tensor
        std::shared_ptr<T[]> data; ///< The shared_ptr to the underlying C array
    public:
        Tensor<T> copy() const; ///< Returns a copy of the tensor (the underlying data is copied)
        void copy(const Tensor<T>& other); ///< Copy the content of another tensor
        Tensor<T> reshape(const std::vector<unsigned int>& new_shape) const; ///< Returns the tensor reshaped (the underlying data is shared)
        Tensor<T> flatten() const; ///< Returns the tensor reshaped into a 1D tensor
        T* address() const; ///< Return the adress of the underlying data
        bool increment(std::vector<unsigned int>& coords) const; ///< Increment the given index, returns false, if out of bounds
        Tensor<unsigned int> index_max(unsigned int axis=0) const; ///< Return the index of the max value along the given axis
        Tensor<T> at_index(const Tensor<unsigned int>& coords) const; ///< Build a tensor from a tensor of index
        static Tensor<T> concatenate(const std::vector<Tensor<T>>& tensors, unsigned int axis=0); ///< Concatenate a list of tensor along the given axis
    public:
        Tensor<T> operator[](unsigned int i) const;
        T& operator[](const std::vector<unsigned int>& coords);
        T operator[](const std::vector<unsigned int>& coords) const;
        Tensor<T> operator+(const Tensor<T>& other) const;
        Tensor<T> operator-(const Tensor<T>& other) const;
        Tensor<T> operator*(const Tensor<T>& other) const;
        Tensor<T> operator/(const Tensor<T>& other) const;
        Tensor<T>& operator+=(const Tensor<T>& other);
        Tensor<T>& operator-=(const Tensor<T>& other);
        Tensor<T>& operator*=(const Tensor<T>& other);
        Tensor<T>& operator/=(const Tensor<T>& other);
        bool operator==(const Tensor<T>& other) const;
        Tensor<T>& operator=(const Tensor<T>& other);
        explicit operator T() const;
        template <typename t>
        friend std::ostream& operator<<(std::ostream& os, const Tensor<t>& other);
    protected:
        void _allocate(); ///< allocate the memory of the array
        void _set_size(); ///< Calculate the size of the vector from the shape
        static void _throw_bad_shape(const std::string& func_name); ///< throw an std::range_error because the shape of two tensors don't match
        static void _throw_bad_indexing(const std::string& func_name); ///< throw an std::range_error because the number of coordinates does'nt match the tensors shape
        unsigned int _coords_to_index(const std::vector<unsigned int>& coords) const; ///< converts an n-D coordinate to the index in the 1D array
        Tensor<T> _expand_to_dim(unsigned int dim) const; ///< Returns a tensor with unit-length axis appended at the front of the shape to reach dimension "dim".
        std::vector<unsigned int> _clamp_coords(const std::vector<unsigned int>&) const; ///< Clamp the coordinates to the shape of the vector
    protected:
        static Tensor<T> _binary_operator(const Tensor<T>& left, const Tensor<T>& right, T (*func)(const T& left, const T& right));
        void _inplace_operator(const Tensor<T>& right, T (*func)(const T& left, const T& right));
        static T _add(const T& left, const T& right);
        static T _sub(const T& left, const T& right);
        static T _mul(const T& left, const T& right);
        static T _div(const T& left, const T& right);
    };

    //Python interface
    extern "C" Tensor<double>* new_Tensor(double* array, unsigned int* shape, unsigned int n_dims);
    extern "C" Tensor<double>* Tensor_copy(Tensor<double>* tensor);
    extern "C" Tensor<double>* Tensor_reshape(Tensor<double>* tensor, unsigned int* new_shape, unsigned int n_dims);
    extern "C" unsigned int Tensor_ndims(Tensor<double>* tensor);
    extern "C" unsigned int* Tensor_shape(Tensor<double>* tensor);
    extern "C" double* Tensor_data(Tensor<double>* tensor);
    extern "C" void Tensor_set(Tensor<double>* tensor, unsigned int* where, double val);
    extern "C" double Tensor_get(Tensor<double>* tensor, unsigned int* where);
    extern "C" Tensor<double>* Tensor_subTensor(Tensor<double>* tensor, unsigned int index);
    extern "C" void del_Tensor(Tensor<double>* tensor);
    extern "C" Tensor<double>* Tensor_concatenate(Tensor<double>** tensors, unsigned int n_tensors, unsigned int axis=0);
}
