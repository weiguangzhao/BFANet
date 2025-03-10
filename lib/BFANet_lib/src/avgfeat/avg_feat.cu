//
// Created by weiguang zhao on 2024-04-17.
//

#include <fstream>
#include <vector>
#include <algorithm>

#include "avg_feat.cuh"
#include "../bfanet/con_component.cuh"
#include "../bfanet/functions.cuh"

using namespace std;


void avg_feat_radius(at::Tensor x_tensor, at::Tensor y_tensor, at::Tensor z_tensor, at::Tensor batch_index_tensor,
                     at::Tensor sem_tensor, at::Tensor feat_in_tensor, at::Tensor feat_out_tensor,
                     at::Tensor l1_norm_tensor, at::Tensor index_mapper_tensor, at::Tensor cluster_id_tensor,
                     int batch_size, float radius, int dim_num) {

    // ======================================get C++ pointer=============================
    float *x = x_tensor.data<float>();
    float *y = y_tensor.data<float>();
    float *z = z_tensor.data<float>();
    float *l1_norm = l1_norm_tensor.data<float>();
    int *index_mapper = index_mapper_tensor.data<int>();


    float *feat_in = feat_in_tensor.data<float>();
    float *feat_out = feat_out_tensor.data<float>();
    int *batch_ind = batch_index_tensor.data<int>();

    int *sem = sem_tensor.data<int>();
    int *cluster_idx = cluster_id_tensor.data<int>(); //cluster index for points


    //====pointers for batch variables
    float *x_s, *y_s, *z_s, *feat_in_s, *feat_out_s, *l1_norm_s;
    int *sem_s, *index_mapper_s, *cluster_idx_s;

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
        l1_norm_s = l1_norm + batch_start;
        index_mapper_s = index_mapper + batch_start;

        feat_in_s = feat_in + batch_start*dim_num;
        feat_out_s = feat_out + batch_start*dim_num;
        sem_s = sem + batch_start;
        cluster_idx_s = cluster_idx + batch_start;

        //====Public variables
        BFANET::Solver solver(x_s, y_s, z_s, l1_norm_s, index_mapper_s,  batch_length, radius, cluster_idx_s, sem_s);
        //====Sort the input by the l1norm of each point.
        solver.sort_input_by_l1norm();
        //====calculate the number of neighbor
        solver.calc_num_neighbours();
        //====Prefix sum.
        solver.calc_start_pos();
        //====populate the actual neighbours for each vertex.
        solver.append_neighbours();

        //========cal the average feat
        solver.avg_feat(feat_in_s, feat_out_s, dim_num);

        batch_start += batch_length;
    }

}
