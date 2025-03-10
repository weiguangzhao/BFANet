//
// Created by weiguang zhao on 2024-04-17.
//

#ifndef BFANET_LIB_AVG_FEAT_CUH
#define BFANET_LIB_AVG_FEAT_CUH

void avg_feat_radius(at::Tensor x_tensor, at::Tensor y_tensor, at::Tensor z_tensor, at::Tensor batch_index_tensor,
                     at::Tensor sem_tensor, at::Tensor feat_in_tensor, at::Tensor feat_out_tensor,
                     at::Tensor l1_norm_tensor, at::Tensor index_mapper_tensor, at::Tensor cluster_id_tensor,
                     int batch_size, float radius, int dim_num);

#endif //BFANET_LIB_AVG_FEAT_CUH
