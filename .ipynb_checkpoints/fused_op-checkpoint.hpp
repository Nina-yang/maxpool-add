#ifndef FUSED_OP_H
#define FUSED_OP_H

#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include <memory>
#include <iostream>
#include <cassert>


#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

#include "tensor.hpp"


void get_idx(const size_t& idx_dst, size_t& idx_a, size_t& idx_b, const Stride& str_dst, const Stride& str_a, const Stride& str_b){
    size_t p_dst, k_dst, i_dst, j_dst;
    size_t p_srca, k_srca, i_srca, j_srca;
    size_t p_srcb, k_srcb, i_srcb, j_srcb; 
    size_t tmp;
    
    tmp = idx_dst;
    p_dst = tmp / str_dst.stride_B;
    tmp %= str_dst.stride_B;
    k_dst = tmp / str_dst.stride_C;
    tmp %= str_dst.stride_C;
    i_dst = tmp / str_dst.stride_H;
    tmp %= str_dst.stride_H;
    j_dst = tmp;
    
    
    p_srca = str_a.B == 1? 0: p_dst;
    k_srca = str_a.C == 1? 0: k_dst;
    i_srca = str_a.H == 1? 0: i_dst;
    j_srca = str_a.W == 1? 0: j_dst;
    idx_a = p_srca * str_a.stride_B + k_srca * str_a.stride_C + i_srca * str_a.stride_H + j_srca;

    p_srcb = str_b.B == 1? 0: p_dst;
    k_srcb = str_b.C == 1? 0: k_dst;
    i_srcb = str_b.H == 1? 0: i_dst;
    j_srcb = str_b.W == 1? 0: j_dst;
    idx_b = p_srcb * str_b.stride_B + k_srcb * str_b.stride_C + i_srcb * str_b.stride_H + j_srcb;
    
//     assert(idx_a<str_a.B*str_a.C*str_a.H*str_a.W);
//     assert(idx_b<str_b.B*str_b.C*str_b.H*str_b.W);
    
//     if(str_b.B != str_a.B || str_b.C != str_a.C){
//         std::cout << "idx_a: " << idx_a << ", " << str_a.B*str_a.C*str_a.H*str_a.W << std::endl;
//         std::cout << "idx_b: " << idx_b << ", " << str_b.B*str_b.C*str_b.H*str_b.W << std::endl;
//         std::cout << std::endl << std::endl;
//     }
}

template<typename T> 
void fused_add_array(T* p_a, T*p_b, T* res, size_t size, Stride& str_dst,  Stride& str_a, Stride& str_b) {
    size_t idx_dst ;
    size_t idx_a, idx_b;

    #ifdef USE_OMP
    #pragma omp parallel shared(p_a, p_b, res) private(idx_dst, idx_a, idx_b) num_threads(4)
    {
        #pragma omp for schedule(static)
    #endif
    for(idx_dst = 0; idx_dst < size; ++idx_dst) {
        get_idx(idx_dst, idx_a, idx_b, str_dst, str_a, str_b);
        res[idx_dst] = p_a[idx_a] + p_b[idx_b];
    }
    #ifdef USE_OMP
    }
    #endif
}

template<typename T>
Tensor<T> fused_add(Tensor<T>& a, Tensor<T>& b) {
    int states = elem_wise_op_size_check(a, b);
    if (states == -1) 
        return Tensor<T>(); // size mismatch, return an invalid tensor indicating elem-wise add cannot be applied to a and b.
    size_t res_B = std::max(a.B, b.B);
    size_t res_C = std::max(a.C, b.C);
    size_t res_H = std::max(a.H, b.H);
    size_t res_W = std::max(a.W, b.W);

    Tensor<T> res(res_B, res_C, res_H, res_W);
    
    Stride str_res = res.stride();
    Stride str_a = a.stride();
    Stride str_b = b.stride();
    
    fused_add_array(a.p, b.p, res.p, res.size(), str_res, str_a, str_b);
    
    return res;
}

#endif // FUSED_OP_H