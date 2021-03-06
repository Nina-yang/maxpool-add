#ifndef MAX_POOL_ADD_H
#define MAX_POOL_ADD_H

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
#include "tensor.hpp"
#include "fused_op.hpp"

int th_num = 4;

template<typename T> 
void add_array(T* p_a, T*p_b, T* res, size_t size) {
    size_t i;

    #ifdef USE_OMP
    #pragma omp parallel shared(p_a, p_b, res) private(i) num_threads(th_num)
    {
        #pragma omp for schedule(static)
    #endif
        for(i = 0; i < size; ++i) {
            res[i] = p_a[i] + p_b[i];
        }
    #ifdef USE_OMP
    }
    #endif

}

#ifdef USE_AVX
template <>
void add_array(int* p_a, int*p_b, int* res, size_t size) {
    size_t i ;
    
    #ifdef USE_OMP
    #pragma omp parallel shared(p_a, p_b, res) private(i) num_threads(th_num)
    {
        #pragma omp for schedule(static)
    #endif
        for( i = 0; i < size; i += 16){
            __m512i A = _mm512_loadu_epi32((const __m512i*) (p_a + i));
            __m512i B = _mm512_loadu_epi32((const __m512i*) (p_b + i));
            __m512i C = _mm512_add_epi32(A, B);
            _mm512_storeu_si512((__m512i*)(res + i), C);
        }
    #ifdef USE_OMP   
    }
    #endif

    i = size - size % 16;
    for(; i < size; ++i) {
        res[i] = p_a[i] + p_b[i];
    }
}


template <>
void add_array(float* p_a, float* p_b, float* res, size_t size) {
    size_t i ;

    #ifdef USE_OMP
    #pragma omp parallel shared(p_a, p_b, res) private(i) num_threads(th_num)
    {
        #pragma omp for schedule(static)
    #endif
        for( i = 0; i <= size -16; i += 16){
            __m512 A = _mm512_loadu_ps((p_a + i));
            __m512 B = _mm512_loadu_ps((p_b + i));
            __m512 C = _mm512_add_ps(A, B);
            _mm512_storeu_si512((res + i), C);
        }   
    #ifdef USE_OMP
    }
    #endif
    i = size - size % 16;
    for(; i < size; ++i) {
        res[i] = p_a[i] + p_b[i];
    }
}

template <>
void add_array(double* p_a, double* p_b, double* res, size_t size) {
    size_t i ;

    #ifdef USE_OMP
    #pragma omp parallel shared(p_a, p_b, res) private(i) num_threads(th_num)
    {
        #pragma omp for schedule(static)
    #endif
        for( i = 0; i <= size -8; i += 8){
            __m512d A = _mm512_loadu_pd((p_a + i));
            __m512d B = _mm512_loadu_pd((p_b + i));
            __m512d C = _mm512_add_pd(A, B);
            _mm512_storeu_si512((res + i), C);
        }   
    #ifdef USE_OMP
    }
    #endif
    i = size - size % 8;
    for(; i < size; ++i) {
        res[i] = p_a[i] + p_b[i];
    }
}
#endif // USE_AVX


template<typename T>
void padding_2D(T* src, T* dst, size_t H, size_t W) {
    size_t i, j;
    #ifdef USE_OMP
    #pragma omp parallel private(i,j) num_threads(th_num)
    {
        #pragma omp for schedule(static)
    #endif
        for(i = 0 ; i < H; ++i) {
            for(j = 0; j < W; ++j) {
                dst[(i + 1) * (W + 2) + j + 1] = src[i * W + j];
            }
        }
    #ifdef USE_OMP
    }
    #endif
}

template<typename T>
void max_pool_2D(T* src, T* dst, size_t src_H, size_t src_W, size_t dst_H, size_t dst_W) {
    T max_elem;
    size_t i,j,k_i,k_j;
    #ifdef USE_OMP
    #pragma omp parallel private(i,j,k_i,k_j,max_elem) num_threads(th_num)
    {
        #pragma omp for schedule(static)
    #endif
        for(i = 0; i < src_H-2;  i += 2) {
            for (j = 0; j < src_W-2; j += 2) {
                max_elem = src[i * src_W + j];
                for(k_i  = 0; k_i < 3; ++k_i) {
                    for(k_j = 0; k_j < 3; ++ k_j) {
                        if (max_elem < src[(i + k_i) * src_W + j + k_j]) {
                            max_elem = src[(i + k_i) * src_W + j + k_j];
                        }
                    }
                }
                dst[(i/2) * dst_W + j/2] = max_elem;
            }
        } 
    #ifdef USE_OMP
    }   
    #endif

}

template<typename T>
void fused_pad_max_pool_2D(T* src, T* dst, size_t src_H, size_t src_W, size_t dst_H, size_t dst_W) {
    T max_elem;
    size_t i,j,k_i,k_j;
    size_t idw_first_padded = 2, idh_first_padded=2;
    size_t idw_last_padded = 2 * ((src_W-1)/2);
    size_t idh_last_padded = 2 * ((src_H-1)/2);
    size_t padded_H = src_H + 2;
    size_t padded_W = src_W + 2;
    
    // std::cout << "Shape: srcH=" << src_H << ",srcW=" << src_W << std::endl;
    // std::cout << "idh_last_padded=" << idh_last_padded << std::endl;
    // std::cout << "idw_last_padded=" << idw_last_padded << std::endl;
    // std::cout << "dst_H=" << dst_H << ",dst_W" << dst_W << std::endl;
    // std::cout << std::endl << std::endl << std::endl;
    
    // #ifdef USE_OMP
    // #pragma omp parallel private(i,j,k_i,k_j,max_elem) num_threads(th_num)
    // {
    //     #pragma omp for schedule(static)
    // #endif
        for(i = idh_first_padded; i < idh_last_padded;  i += 2) {
            for (j = idw_first_padded; j < idw_last_padded; j += 2) {
                max_elem = src[(i-1) * src_W + (j-1)];
                for(k_i  = 0; k_i < 3; ++k_i) {
                    for(k_j = 0; k_j < 3; ++ k_j) {
                        if (max_elem < src[(i - 1 + k_i) * src_W + j -1 + k_j]) {
                            max_elem = src[(i - 1 + k_i) * src_W + j -1 + k_j];
                        }
                    }
                }
                dst[(i/2) * dst_W + j/2] = max_elem;
            }
        } 
    // #ifdef USE_OMP
    // }   
    // #endif
    
    // upper slot
    for(i=0; i<idh_first_padded; i+=2){
        for(j=0;j<padded_W-2; j+=2){
            max_elem = 0;
            // max_elem = src[i * src_W + j];
            for(k_i  = 0; k_i < 3; ++k_i) {
                for(k_j = 0; k_j < 3; ++ k_j) {
                    if(i-1+k_i < 0 || j - 1 + k_j < 0 || i-1+k_i >= src_H || j - 1 + k_j >= src_W) continue;
                    if (max_elem < src[(i-1 + k_i) * src_W + j - 1 + k_j]) {
                        max_elem = src[(i-1 + k_i) * src_W + j - 1 + k_j];
                    }
                }
            }
            dst[(i/2) * dst_W + j/2] = max_elem;
        }
    }
    
    // lower slot
    for(i=idh_last_padded; i<padded_H-2; i+=2){
        for(j=0;j<padded_W-2; j+=2){
            max_elem = 0;
            for(k_i  = 0; k_i < 3; ++k_i) {
                for(k_j = 0; k_j < 3; ++ k_j) {
                    if(i-1+k_i < 0 || j - 1 + k_j < 0 || i-1+k_i >= src_H || j - 1 + k_j >= src_W) continue;
                    if (max_elem < src[(i-1 + k_i) * src_W + j - 1 + k_j]) {
                        max_elem = src[(i-1 + k_i) * src_W + j - 1 + k_j];
                    }
                }
            }
            dst[(i/2) * dst_W + j/2] = max_elem;
        }
    }
    
    // left slot
    for(i=0; i<padded_H-2; i+=2){
        for(j=0;j<idw_first_padded; j+=2){
            max_elem = 0;
            for(k_i  = 0; k_i < 3; ++k_i) {
                for(k_j = 0; k_j < 3; ++ k_j) {
                    if(i-1+k_i < 0 || j - 1 + k_j < 0 || i-1+k_i >= src_H || j - 1 + k_j >= src_W) continue;
                    if (max_elem < src[(i-1 + k_i) * src_W + j - 1 + k_j]) {
                        max_elem = src[(i-1 + k_i) * src_W + j - 1 + k_j];
                    }
                }
            }
            dst[(i/2) * dst_W + j/2] = max_elem;
        }
    }
    
    // right slot
    for(i=0; i<padded_H-2; i+=2){
        for(j=idw_last_padded;j<padded_W-2; j+=2){
            max_elem = 0;
            for(k_i  = 0; k_i < 3; ++k_i) {
                for(k_j = 0; k_j < 3; ++ k_j) {
                    if(i-1+k_i < 0 || j - 1 + k_j < 0 || i-1+k_i >= src_H || j - 1 + k_j >= src_W) continue;
                    if (max_elem < src[(i-1 + k_i) * src_W + j - 1 + k_j]) {
                        max_elem = src[(i-1 + k_i) * src_W + j - 1 + k_j];
                    }
                }
            }
            dst[(i/2) * dst_W + j/2] = max_elem;
        }
    }

}

template<typename T>
Tensor<T> max_pool(Tensor<T>& src) {
    // pre-allocated memory for holding all the following padded 2-D matrix.
    // all the elements are initialized as 0 to save the ???0 assignment" in default padding.
    // T* padding = (T*) calloc((src.H + 2) * (src.W + 2), sizeof(T));
    T* padding = (T*) calloc(src.B * src.C * (src.H + 2) * (src.W + 2), sizeof(T));
    size_t stride_B = src.C * (src.H + 2) * (src.W + 2);
    size_t stride_C = (src.H + 2) * (src.W + 2);
    if (padding == nullptr) {
        abort();
    }

    size_t dst_H =  (src.H + 1) / 2;
    size_t dst_W =  (src.W + 1) / 2 ;
    Tensor<T> dst(src.B, src.C, dst_H, dst_W);

    Stride src_str = src.stride();
    Stride dst_str=  dst.stride();

    size_t b_i, c_i;
    T* current_padding;
    
    #ifdef USE_BATCHOMP
    #pragma omp parallel shared(padding, dst) private(b_i,c_i,current_padding) num_threads(5)
    {
        #pragma omp for schedule(dynamic)
    #endif
    
    for (b_i = 0 ; b_i < dst.B; ++b_i) {
        for(c_i = 0; c_i < dst.C; ++c_i) {
            #ifdef USE_OP_FUSION
            fused_pad_max_pool_2D(src.p + b_i * src_str.stride_B + c_i * src_str.stride_C, 
                                  dst.p + b_i * dst_str.stride_B + c_i * dst_str.stride_C, 
                                  src.H, src.W, dst_H, dst_W);
            #else
            current_padding = padding + b_i * stride_B + c_i * stride_C;
            padding_2D(src.p + b_i * src_str.stride_B + c_i * src_str.stride_C, current_padding, src.H, src.W); 
            max_pool_2D(current_padding, dst.p + b_i * dst_str.stride_B + c_i * dst_str.stride_C, src.H + 2, src.W + 2, dst_H, dst_W);
            #endif
        }
    }
        
    #ifdef USE_BATCHOMP
    }
    #endif

    free(padding);

    return dst;
}



/**
 * @brief 
 * 
 * @return 
 * -1: not support elem-wise operation. 
 * 0: no broadcast requirement
 * 1: broadcast on the first tensor
 * 2: broadcast on the second tensor
 * 3: broadcast on the both tensors.
 *      
 */
template<typename T> 
int elem_wise_op_size_check(Tensor<T>& a, Tensor<T> & b) { 
    if (a.B != 1 && b.B != 1 && a.B != b.B) return -1;
    if (a.C != 1 && b.C != 1 && a.C != b.C) return -1;
    if (a.H != 1 && b.H != 1 && a.H != b.H) return -1;
    if (a.W != 1 && b.W != 1 && a.W != b.W) return -1;
    
    int broadcast_cnt = 0;
    if (a.B == 1 || a.C == 1 || a.H == 1 || a.W == 1) broadcast_cnt += 1;
    if (b.B == 1 || b.C == 1 || b.H == 1 || b.W == 1) broadcast_cnt += 2;
    return broadcast_cnt;
}


template<typename T>
void expand(Tensor<T>& src, Tensor<T>& dst) { // expand for broadcast,  e.g., 4*1*200*1 -> 4*3*200*400 
    Stride src_str = src.stride();
    Stride dst_str = dst.stride();
    size_t src_index;
    size_t b_i, c_i, h_i, w_i;
    
    // adjust strides of dimensions with 1 to 0 for broadcasting.
    src_str.stride_B *= (src.B == 1 ? 0 : 1);
    src_str.stride_C *= (src.C == 1 ? 0 : 1);
    src_str.stride_H *= (src.H == 1 ? 0 : 1);
    src_str.stride_W *= (src.W == 1 ? 0 : 1);

    #ifdef USE_OMP
    #pragma omp parallel private(b_i,c_i,h_i,w_i, src_index) num_threads(th_num)
    {
        #pragma omp for schedule(static)
    #endif
        for(b_i = 0; b_i < dst.B; ++b_i) {
            for(c_i = 0; c_i < dst.C; ++c_i) {
                for(h_i = 0; h_i < dst.H; ++h_i) {
                    for(w_i = 0; w_i < dst.W; ++ w_i) {
                        src_index = 0;
                        src_index += b_i * src_str.stride_B;
                        src_index += c_i * src_str.stride_C;
                        src_index += h_i * src_str.stride_H;
                        src_index += w_i * src_str.stride_W;
                        dst.p[b_i * dst_str.stride_B + c_i * dst_str.stride_C + h_i * 
                        dst_str.stride_H + w_i * dst_str.stride_W] = src.p[src_index];
                    }
                }
            }
        }
    #ifdef USE_OMP
    }
    #endif
}


template<typename T>
Tensor<T> add(Tensor<T>& a, Tensor<T>& b) {
    int states = elem_wise_op_size_check(a, b);
    if (states == -1) 
        return Tensor<T>(); // size mismatch, return an invalid tensor indicating elem-wise add cannot be applied to a and b.
    size_t res_B = std::max(a.B, b.B);
    size_t res_C = std::max(a.C, b.C);
    size_t res_H = std::max(a.H, b.H);
    size_t res_W = std::max(a.W, b.W);

    Tensor<T> res(res_B, res_C, res_H, res_W);

    if (states == 0) {  // no broadcast
        add_array(a.p, b.p, res.p, res.size());
    }else if (states == 1) { // broadcast a 
        Tensor<T> broadcast_a(res_B, res_C, res_H, res_W);
        expand(a, broadcast_a);
        add_array(broadcast_a.p, b.p, res.p, res.size());
    } else if (states == 2) { // broadcast b 
        Tensor<T> broadcast_b(res_B, res_C, res_H, res_W);
        expand(b, broadcast_b);
        add_array(a.p, broadcast_b.p, res.p, res.size());
    } else if (states == 3) { // broadcast a and b
        Tensor<T> broadcast_a(res_B, res_C, res_H, res_W);
        expand(a, broadcast_a);
        Tensor<T> broadcast_b(res_B, res_C, res_H, res_W);
        expand(b, broadcast_b);
        add_array(broadcast_a.p, broadcast_b.p, res.p, res.size());
    }
    return res;
}

template<typename T>
Tensor<T> max_pool_add(Tensor<T>& a, Tensor<T>& b) {
    Tensor<T> a_max_pool = max_pool(a);
    #ifdef USE_OP_FUSION
    Tensor<T> res = fused_add(a_max_pool, b);
    #else
    Tensor<T> res = add(a_max_pool, b);
    #endif
    
    return res;
}

template <typename T>
void helper_fill_sequence(Tensor<T> & tensor) {
    size_t cnt = 0;
    size_t tensor_len = tensor.size();
    for(size_t i = 0; i < tensor_len; ++i) {
        tensor.p[i] = cnt;
        cnt += 1;
    }
}

//
#endif // MAX_POOL_ADD_H
