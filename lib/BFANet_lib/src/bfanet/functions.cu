#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <math.h>
#include "functions.cuh"

constexpr int kSharedMemBytes = 48 * 1024;

/*!
 * Calculate the number of neighbours of each vertex. One kernel thread per
 * vertex.
 * @param x - x values, sorted by l1 norm.
 * @param y - y values, sorted by l1 norm.
 * @param z - z values, sorted by l1 norm.
 * @param l1norm - sorted l1 norm.
 * @param vtx_mapper - maps sorted vertex index to original.
 * @param rad - radius.
 * @param num_vtx - number of vertices.
 * @param num_nbs - output array.
 */
__global__ void k_num_nbs(float const *const x, float const *const y, float const *const z,
                          float const *const l1norm, int const *const vtx_mapper, float radius_,
                          const int num_vtx, int *const num_nbs) {
    int const thread_index = blockIdx.x * blockDim.x + threadIdx.x ;
    if (thread_index >= num_vtx) return;

    // first vtx of current block.
    const int tb_start = blockIdx.x * blockDim.x;
    // last vtx of current block.
    const int tb_end = min(tb_start + blockDim.x, num_vtx) - 1;

    int land_id = threadIdx.x & 0x1f;
    float const *possible_range_start, *possible_range_end;
    if (land_id == 0) {
        // inclusive start
        // https://github.com/NVIDIA/thrust/issues/1734
        possible_range_start = thrust::lower_bound(thrust::seq, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * radius_);
        // exclusive end
        possible_range_end = thrust::upper_bound(thrust::seq, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * radius_);
    }
    possible_range_start =(float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_start, 0);
    possible_range_end =(float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_end, 0);

    // the number of threads might not be blockDim.x, if this is the last block.
    int const num_threads = tb_end - tb_start + 1;
    const int tile_size = kSharedMemBytes / 4 / (1 + 1 + 1);
    // first half of shared stores Xs; second half stores Ys; third half stores Ys.
    __shared__ float shared[tile_size * (1 + 1 + 1)];
    auto *const sh_x = shared;
    auto *const sh_y = shared + tile_size;
    auto *const sh_z = shared + tile_size*2;
    int ans = 0;

    for (auto curr_ptr = possible_range_start; curr_ptr < possible_range_end;
         curr_ptr += tile_size) {
        // curr_ptr's index
        int const curr_idx = curr_ptr - l1norm;
        // current range; might be less than tile_size.
        int const curr_range = min(tile_size, static_cast<int>(possible_range_end - curr_ptr));
        // thread 0 updates sh_x[0], sh_x[0+num_threads], sh_x[0+2*num_threads] ...
        // thread 1 updates sh_x[1], sh_x[1+num_threads], sh_x[1+2*num_threads] ...
        // ...
        // thread t updates sh_x[t], sh_x[t+num_threads], sh_x[t+2*num_threads] ...
        __syncthreads();
        for (auto i = threadIdx.x; i < curr_range; i += num_threads) {
            sh_x[i] = x[curr_idx + i];
            sh_y[i] = y[curr_idx + i];
            sh_z[i] = z[curr_idx + i];
        }
        __syncthreads();
        const float thread_x = x[thread_index], thread_y = y[thread_index], thread_z = z[thread_index];
        for (auto j = 0; j < curr_range; ++j) {
            ans += square_dist(thread_x, thread_y, thread_z, sh_x[j], sh_y[j], sh_z[j]) <=radius_ * radius_;
        }
    }
    num_nbs[vtx_mapper[thread_index]] = ans - 1;
}
/*!
 * Populate the neighbours array. One kernel thread per vertex.
 * @param x - x values, sorted by l1 norm.
 * @param y - y values, sorted by l1 norm.
 * @param l1norm - sorted l1 norm.
 * @param vtx_mapper - maps sorted vertex index to original.
 * @param start_pos - neighbours starting index of each vertex.
 * @param rad - radius.
 * @param num_vtx - number of vertices
 * @param neighbours - output array
 */
__global__ void k_append_neighbours(float const *const x, float const *const y, float const *const z,
                                    float const *const l1norm,
                                    int const *const vtx_mapper,
                                    int const *const start_pos,
                                    float radius_, const int num_vtx,
                                    int *const neighbours) {
    int const thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index >= num_vtx) return;

    // first vtx of current block.
    const int tb_start = blockIdx.x * blockDim.x;
    // last vtx of current block.
    const int tb_end = min(tb_start + blockDim.x, num_vtx) - 1;

    int land_id = threadIdx.x & 0x1f;
    float const *possible_range_start, *possible_range_end;
    if (land_id == 0) {
        // inclusive start
        possible_range_start = thrust::lower_bound(thrust::seq, l1norm, l1norm + num_vtx, l1norm[tb_start] - 2 * radius_);
        // exclusive end
        possible_range_end = thrust::upper_bound(thrust::seq, l1norm, l1norm + num_vtx, l1norm[tb_end] + 2 * radius_);
    }
    possible_range_start =(float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_start, 0);
    possible_range_end =(float *)__shfl_sync(0xffffffff, (uint64_t)possible_range_end, 0);

    int const num_threads = tb_end - tb_start + 1;
    // different from previous kernel, here the shared array is tri-partitioned,
    // because of the frequent access to vtx_mapper.
    const int tile_size = kSharedMemBytes / 4 / (1 + 1 + 1 + 1);
    __shared__ float shared[tile_size * (1 + 1 + 1 + 1)];
    auto *const sh_x = shared;
    auto *const sh_y = shared + tile_size;
    auto *const sh_z = shared + tile_size*2;
    auto *const sh_vtx_mapper = (int *)(sh_z + tile_size);
    int upos = start_pos[vtx_mapper[thread_index]];

    for (auto curr_ptr = possible_range_start; curr_ptr < possible_range_end; curr_ptr += tile_size) {
        // curr_ptr's index
        int const curr_idx = curr_ptr - l1norm;
        // current range; might be less than tile_size.
        int const curr_range =min(tile_size, static_cast<int>(possible_range_end - curr_ptr));
        // thread 0 updates sh_x[0], sh_x[0+num_threads], sh_x[0+2*num_threads] ...
        // thread 1 updates sh_x[1], sh_x[1+num_threads], sh_x[1+2*num_threads] ...
        // ...
        // thread t updates sh_x[t], sh_x[t+num_threads], sh_x[t+2*num_threads] ...
        __syncthreads();
        for (auto i = threadIdx.x; i < curr_range; i += num_threads) {
            sh_x[i] = x[curr_idx + i];
            sh_y[i] = y[curr_idx + i];
            sh_z[i] = z[curr_idx + i];
            sh_vtx_mapper[i] = vtx_mapper[curr_idx + i];
        }
        __syncthreads();
        const float thread_x = x[thread_index], thread_y = y[thread_index], thread_z = z[thread_index];
        for (auto j = 0; j < curr_range; ++j) {
            if (thread_index != curr_idx + j && square_dist(thread_x, thread_y, thread_z,
                                                            sh_x[j], sh_y[j], sh_z[j]) <= radius_ * radius_) {
                neighbours[upos++] = sh_vtx_mapper[j];
            }
        }
    }
}

/*!
 * Traverse the graph from each vertex. One kernel thread per vertex.
 * @param visited - boolean array that tracks if a vertex has been visited.
 * @param frontier - boolean array that tracks if a vertex is on the frontier.
 * @param num_nbs - the number of neighbours of each vertex.
 * @param start_pos - neighbours starting index of each vertex.
 * @param neighbours - the actually neighbour indices of each vertex.
 * @param num_vtx - number of vertices of the graph.
 */
__global__ void k_bfs(bool *const visited, bool *const frontier,
                      int const *const num_nbs,
                      int const *const start_pos,
                      int const *const neighbours,
                      int *dev_sem_,
                      int sem_cur,
                      int num_vtx) {
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= num_vtx) return;
    if (!frontier[u]) return;
    frontier[u] = false;
    visited[u] = true;
    int u_start = start_pos[u];
    for (int i = 0; i < num_nbs[u]; ++i) {
        int v = neighbours[u_start + i];
        if (dev_sem_[v]==sem_cur && !visited[v])  frontier[v] = true;
    }
}

__device__ inline float square_dist(const float x1, const float y1, const float z1,
                                    const float x2, const float y2, const float z2) {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2);
}

__global__ void get_cc_id(bool *const dev_visited, int *dev_cluster_idx_, int *dev_sem_, int cluster, int sem_cur,
                          int num_vtx){
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u >= num_vtx) return;
    if(dev_visited[u] && dev_sem_[u] == sem_cur) dev_cluster_idx_[u] = cluster;
}


__global__ void search_mean_feat(float *dev_feat_in, float *dev_feat_out,
                                 int const *const num_nbs,
                                 int const *const start_pos,
                                 int const *const neighbours,
                                 int *dev_sem_, int num_vtx_, int dim_num){
    int const u = threadIdx.x + blockIdx.x * blockDim.x;
    if (u>num_vtx_) return;

    int cur_sem = dev_sem_[u];
    int u_start = start_pos[u];
    int cur_num_nbs = num_nbs[u];
    int count=0.0;

    for (int i = 0; i < cur_num_nbs; ++i) {
        int v = neighbours[u_start + i];
        if (dev_sem_[v]==cur_sem)  {
            for (int j=0; j<dim_num; j++){
                dev_feat_out[u*dim_num + j] = dev_feat_out[u*dim_num + j]/(count+1.0) * count + dev_feat_in[v*dim_num + j]/(count+1.0);
            }
            count = count + 1.0;
        }
    }

    if (count == 0.0){
        for(int j=0; j<dim_num; j++){
            dev_feat_out[u*dim_num +j] = dev_feat_in[u*dim_num +j];
        }
    }

}
