#ifndef BFANET_CLUSTER_H
#define BFANET_CLUSTER_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void bfanet_cluster(at::Tensor x_tensor, at::Tensor y_tensor, at::Tensor z_tensor, at::Tensor l1_norm_tensor,
                 at::Tensor index_mapper_tensor, at::Tensor batch_index_tensor, at::Tensor sem_tensor,
                 at::Tensor cluster_index_tensor, at::Tensor cluster_num_tensor, at::Tensor clt_sem_tensor,
                 int batch_size, float radius);

#endif //BFANET_CLUSTER_H
