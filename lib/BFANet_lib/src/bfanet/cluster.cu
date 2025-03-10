#include <fstream>
#include <vector>
#include <algorithm>

#include "con_component.cuh"
#include "cluster.h"

using namespace std;
void bfanet_cluster(at::Tensor x_tensor, at::Tensor y_tensor, at::Tensor z_tensor, at::Tensor l1_norm_tensor,
                 at::Tensor index_mapper_tensor, at::Tensor batch_index_tensor, at::Tensor sem_tensor,
                 at::Tensor cluster_index_tensor, at::Tensor cluster_num_tensor, at::Tensor clt_sem_tensor,
                 int batch_size, float radius) {

    // ======================================get C++ pointer=============================
    //=====for kd tree  (offset xyz)
    float *x = x_tensor.data<float>();
    float *y = y_tensor.data<float>();
    float *z = z_tensor.data<float>();
    float *l1_norm = l1_norm_tensor.data<float>();
    int *index_mapper = index_mapper_tensor.data<int>();
    int *sem = sem_tensor.data<int>();

    //====cluster information
    int *batch_ind = batch_index_tensor.data<int>();     //batch division
    //=====what we want to get
    int *cluster_idx = cluster_index_tensor.data<int>(); //cluster index for points
    int *cluster_num = cluster_num_tensor.data<int>();  //cluster number for batch

    //====pointers for batch variables
    float *x_s, *y_s, *z_s, *l1_norm_s;
    int *sem_s, *index_mapper_s, *cluster_idx_s;

    int batch_length = 0;
    int batch_start = 0;
    int cluster_accum = 0;
    int cluster_accum_old = 0;

    //====vector for cluster semantic
    vector<int> clt_sem_vector;
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

        sem_s = sem + batch_start;
        cluster_idx_s = cluster_idx + batch_start;

        //==================BFANET cluster==========================
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

        //==== Cluster HPs
        cluster_accum = solver.identify_clusters(cluster_accum_old, clt_sem_vector);

        cluster_num[batch_i] = cluster_accum - cluster_accum_old;
        if(cluster_num[batch_i] == 0) {
            //printf("batch %d has none cluster \n", batch_i);
            cluster_accum_old = cluster_accum;
            batch_start += batch_length;
            continue;
        }

        //====refresh==============
        cluster_accum_old = cluster_accum;
        batch_start += batch_length;
    }
    //========================output resize=======================
    clt_sem_tensor.resize_({((int)clt_sem_vector.size())});
    int *clt_sem = clt_sem_tensor.data<int>();
    memcpy(clt_sem, &clt_sem_vector[0], clt_sem_vector.size()*sizeof(int ));
}