#ifndef FUSED_GRAPH_H
#define FUSED_GRAPH_H

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


template<typename T>
void fused_max_pool_add_2D(T* src, T* src_b, T* dst, 
                           size_t src_H, size_t src_W,
                           size_t dst_H, size_t dst_W, 
                           size_t flag_h, size_t flag_w);


template<typename T>
void fused_graph(Tensor<T>& srca, Tensor<T>& srcb, Tensor<T>& dst,
                     size_t flag_b, size_t flag_c, size_t flag_h, size_t flag_w) {

    size_t dst_H =  (srca.H + 1) / 2;
    size_t dst_W =  (srca.W + 1) / 2 ;

    Stride srca_str = srca.stride();
    Stride srcb_str = srcb.stride();
    Stride dst_str=  dst.stride();

    size_t b_i, c_i;
    
    #pragma omp parallel shared(srca, srcb, dst) private(b_i,c_i) num_threads(4)
    {
        #pragma omp for schedule(static)
    
    for (b_i = 0 ; b_i < dst.B; ++b_i) {
        for(c_i = 0; c_i < dst.C; ++c_i) {
            fused_max_pool_add_2D(srca.p + b_i * srca_str.stride_B + c_i * srca_str.stride_C, 
                                  srcb.p + b_i * srcb_str.stride_B * flag_b + c_i * srcb_str.stride_C * flag_c,
                                  dst.p + b_i * dst_str.stride_B + c_i * dst_str.stride_C, 
                                  srca.H, srca.W, dst_H, dst_W,
                                 flag_h, flag_w);
        }
    }
        
    }
    
}


template<typename T>
void fused_max_pool_add_2D(T* src, T* src_b, T* dst, 
                           size_t src_H, size_t src_W,
                           size_t dst_H, size_t dst_W, 
                           size_t flag_h, size_t flag_w) {
    T max_elem;
    size_t i,j,k_i,k_j;
    size_t idw_first_padded = 2, idh_first_padded=2;
    size_t idw_last_padded = 2 * ((src_W-1)/2);
    size_t idh_last_padded = 2 * ((src_H-1)/2);
    size_t padded_H = src_H + 2;
    size_t padded_W = src_W + 2;
    

    for(i = idh_first_padded; i < idh_last_padded;  i += 2) {
        for (j = idw_first_padded; j < idw_last_padded; j += 2) {
            max_elem = src[(i-1) * src_W + (j-1)];
            for(k_i  = 0; k_i < 3; ++k_i) {
                for(k_j = 0; k_j < 3; ++ k_j) {
                    if(k_i == 0 && k_j ==0)
                        continue;
                    if (max_elem < src[(i - 1 + k_i) * src_W + j -1 + k_j]) {
                        max_elem = src[(i - 1 + k_i) * src_W + j -1 + k_j];
                    }
                }
            }
            dst[(i/2) * dst_W + j/2] = max_elem + src_b[(i/2) * dst_W * flag_h+ j/2 * flag_w];
        }
    } 

    
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
            dst[(i/2) * dst_W + j/2] = max_elem + src_b[(i/2) * dst_W * flag_h+ j/2 * flag_w];
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
            dst[(i/2) * dst_W + j/2] = max_elem + src_b[(i/2) * dst_W * flag_h+ j/2 * flag_w];
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
            dst[(i/2) * dst_W + j/2] = max_elem + src_b[(i/2) * dst_W * flag_h+ j/2 * flag_w];
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
            dst[(i/2) * dst_W + j/2] = max_elem + src_b[(i/2) * dst_W * flag_h+ j/2 * flag_w];
        }
    }
}

#endif // FUSED_GRAPH_H