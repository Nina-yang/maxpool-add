#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include <memory>
#include <iostream>


#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include "utils.hpp"

struct Stride {
    size_t stride_B;
    size_t stride_C;
    size_t stride_H;
    size_t stride_W;
    
    size_t B;
    size_t C;
    size_t H;
    size_t W;

};

template<typename T>
T* read_data_from_file(const char * path, T* shape);

template<typename T>
struct Tensor {
    static_assert(std::is_same<T, float>::value 
    || std::is_same<T, double>::value 
    || std::is_same<T, int>::value 
    , "Tensor value types are restricted to double, float or int!");

    /**
     * @brief Default constructor, create a tensor that is not valid / initialized
     * 
     */
    Tensor() { }

    Tensor(size_t B, size_t C, size_t H, size_t W) : 
    B(B), C(C), H(H), W(W) {
         p = (T*)malloc(sizeof(T) * B * C * H * W);
    }

    /**
     * @brief Construct a new Tensor object by deserializing data from file
     * 
     * @param path: path of file from which to deserialize
     */
    Tensor(const char* path) { 
        T shape[4];
        p = read_data_from_file(path, shape);
        B = shape[0];
        C = shape[1];
        H = shape[2];
        W = shape[3];
    }


    Tensor(const Tensor<T>& t) { // copy constructor
        B = t.B;
        C = t.C;
        H = t.H;
        W = t.W;
        size_t size = t.size();
        T* p = (T*) malloc(sizeof(T) * size);
        memcpy(p, t.p, sizeof(T) * size);
    }

    Tensor(Tensor<T> && t) { // move constructor
        B = t.B;
        C = t.C;
        H = t.H;
        W = t.W;
        if (p) delete[] p;
        p = t.p;
        t.p = nullptr;
    }

    ~Tensor() {
        if (p) {
            free(p);
        }
    }

    bool operator == (const Tensor<T>& t) const {
        if (B != t.B) return false;
        if (C != t.C) return false;
        if (H != t.H) return false;
        if (W != t.W) return false;
        size_t size = this->size();
        for (size_t i = 0; i < size; ++i) {
            if (p[i] != t.p[i]) 
                return false;
        }  
        return true;

    }

    size_t size() const{
        return B * C * H * W;
    }

    Stride stride() const {
        return Stride{C*H*W, H*W, W, 1, B, C, H, W};
    }

    void print_elems() const {
        size_t size = this->size();
        for( size_t i = 0; i < size; ++i) {
            std::cout << p[i] << std::endl;
        }
    }

    bool is_valid() const {
        return p != nullptr;
    }

    size_t B;
    size_t C;
    size_t H;
    size_t W;
    T* p = nullptr;
};

template<typename T>
T* read_data_from_file(const char * path, T* shape) {
    std::ifstream input;
    input.open(path, std::ios::in | std::ios::binary);
    input.read((char*)shape, 4 * sizeof(T));
    size_t size =(size_t)shape[0] * (size_t)shape[1] * (size_t)shape[2] *(size_t) shape[3]; // cast to size_t to avoid overflow
    T * arr = (T *)malloc(size * sizeof(T));
    input.read((char*)arr, size * sizeof(T));
    input.close();
    return arr;
}

#endif