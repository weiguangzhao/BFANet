//
// Created by weiguang zhao on 2024-01-02.
//
#include <fstream>
#include <vector>
#include <algorithm>

#include "sem_margin.h"
#include "../bfanet/con_component.cuh"
#include "../bfanet/functions.cuh"

using namespace std;

__global__ void run_detect(float *dev_x_, float *dev_y_, float *dev_z_, int *dev_sem_, int *margin_s_dev,
                           float radius, int batch_length){
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= batch_length) return;
    float cur_dist = 0.0;
    int cur_sem = dev_sem_[u];
    if (cur_sem==-100) return ;
    int cor_sem = 0;
    for (int i=0; i<batch_length; i++){
        cor_sem = dev_sem_[i];
        if (cor_sem==-100) continue;
        cur_dist = square_dist(dev_x_[u], dev_y_[u], dev_z_[u], dev_x_[i], dev_y_[i], dev_z_[i]);
        cur_dist = square_dist(dev_x_[u], dev_y_[u], dev_z_[u], dev_x_[i], dev_y_[i], dev_z_[i]);
        if (cor_sem!=cur_sem && cur_dist < radius*radius) margin_s_dev[u]=1;
    }
}

void sem_margin_det(at::Tensor x_tensor, at::Tensor y_tensor, at::Tensor z_tensor, at::Tensor sem_tensor,
                    at::Tensor batch_index_tensor, at::Tensor margin_tensor, float radius, int batch_size){
    // ======================================get C++ pointer=============================
    //====original xyz
    float *x = x_tensor.data<float>();
    float *y = y_tensor.data<float>();
    float *z = z_tensor.data<float>();
    //====other information
    int *sem = sem_tensor.data<int>();                  //semantic info for each point
    int *batch_ind = batch_index_tensor.data<int>();     //batch division
    //=====what we want to get
    int *margin = margin_tensor.data<int>();

    //====pointers for batch variables
    float *x_s, *y_s, *z_s;
    int  *sem_s, *margin_s;
    float *dev_x_s, *dev_y_s, *dev_z_s;
    int  *dev_sem_s, *dev_margin_s;

    int batch_length = 0;
    int batch_start = 0;
    // ======================================cluster the points for each batch=============================
    for(int batch_i=0; batch_i<batch_size; batch_i++){
        batch_length = batch_ind[batch_i];
        if (batch_length==0) {
            continue;
        }
        // =====pointer offset for batch
        x_s = x + batch_start;
        y_s = y + batch_start;
        z_s = z + batch_start;
        sem_s = sem + batch_start;
        margin_s = margin + batch_start;

        // Malloc the CUDA ROOM
        CUDA_ERR_CHK(cudaMalloc((void **) &dev_x_s, sizeof(float) * batch_length));
        CUDA_ERR_CHK(cudaMalloc((void **) &dev_y_s, sizeof(float) * batch_length));
        CUDA_ERR_CHK(cudaMalloc((void **) &dev_z_s, sizeof(float) * batch_length));
        CUDA_ERR_CHK(cudaMalloc((void **) &dev_sem_s, sizeof(int) * batch_length));
        CUDA_ERR_CHK(cudaMalloc((void **) &dev_margin_s, sizeof(int) * batch_length));

        // cudaMemcpyHostToDevice
        CUDA_ERR_CHK(cudaMemcpy(dev_x_s, x_s, sizeof(float) * batch_length, cudaMemcpyHostToDevice));
        CUDA_ERR_CHK(cudaMemcpy(dev_y_s, y_s, sizeof(float) * batch_length, cudaMemcpyHostToDevice));
        CUDA_ERR_CHK(cudaMemcpy(dev_z_s, z_s, sizeof(float) * batch_length, cudaMemcpyHostToDevice));
        CUDA_ERR_CHK(cudaMemcpy(dev_sem_s, sem_s, sizeof(int) * batch_length, cudaMemcpyHostToDevice));
        CUDA_ERR_CHK(cudaMemcpy(dev_margin_s, margin_s, sizeof(int) * batch_length, cudaMemcpyHostToDevice));

        // define block and threads
        dim3 block_n(DIVUP(batch_length, THREADS_PER_BLOCK));
        dim3 threads_n(THREADS_PER_BLOCK);

        run_detect<<<block_n, threads_n>>>(dev_x_s, dev_y_s, dev_z_s, dev_sem_s, dev_margin_s, radius, batch_length);

        CUDA_ERR_CHK(cudaMemcpy(margin_s, dev_margin_s, sizeof(int) * batch_length, cudaMemcpyDeviceToHost));

        // Free CUDA ROOM
        CUDA_ERR_CHK(cudaFree(dev_x_s));
        CUDA_ERR_CHK(cudaFree(dev_y_s));
        CUDA_ERR_CHK(cudaFree(dev_z_s));
        CUDA_ERR_CHK(cudaFree(dev_sem_s));
        CUDA_ERR_CHK(cudaFree(dev_margin_s));

        // set the pointer start address
        batch_start += batch_length;
    }
}