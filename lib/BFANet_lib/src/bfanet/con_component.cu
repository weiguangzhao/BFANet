#include <thrust/count.h>
#include <fstream>
#include <vector>
#include <algorithm>

#include "con_component.cuh"
#include "functions.cuh"

using namespace std;

//================================================= define input style==============================
BFANET::Solver::Solver(float *x_tensor, float *y_tensor, float *z_tensor,float *l1_norm_tensor, int *index_mapper_tensor,
                    int num, float radius, int *cluster_id_tensor, int *sem_tensor){
    num_vtx_= num;
    radius_ = radius;
    num_blocks_ = std::ceil(num_vtx_ / static_cast<float>(BLOCK_SIZE));

    x_ = x_tensor;
    y_ = y_tensor;
    z_ = z_tensor;
    l1norm_ = l1_norm_tensor;
    vtx_mapper_ = index_mapper_tensor;

    sem_ = sem_tensor;
    cluster_ids = cluster_id_tensor;
}

void BFANET::Solver::sort_input_by_l1norm() {
  const auto N = sizeof(x_[0]) * num_vtx_;
  const auto K = sizeof(dev_vtx_mapper_[0]) * num_vtx_;

  CUDA_ERR_CHK(cudaMalloc((void **)&dev_x_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_y_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_z_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_l1norm_, N));
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_vtx_mapper_, K));
  CUDA_ERR_CHK(cudaMemcpy(dev_x_, x_, N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_y_, y_, N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_z_, z_, N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_l1norm_, l1norm_, N, H2D));
  CUDA_ERR_CHK(cudaMemcpy(dev_vtx_mapper_, vtx_mapper_, K, H2D));

  // https://thrust.github.io/doc/classthrust_1_1zip__iterator.html
  typedef typename thrust::tuple<float *, float *, float *, int *> IteratorTuple;
  typedef typename thrust::zip_iterator<IteratorTuple> ZipIterator;
  ZipIterator begin(thrust::make_tuple(dev_x_, dev_y_, dev_z_, dev_vtx_mapper_));
  thrust::sort_by_key(thrust::device, dev_l1norm_, dev_l1norm_ + num_vtx_, begin);
}

void BFANET::Solver::calc_num_neighbours() {
  const auto K = sizeof(dev_num_neighbours_[0]) * num_vtx_;

  int last_vtx_num_nbs = 0;
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_num_neighbours_, K));

  k_num_nbs<<<num_blocks_, BLOCK_SIZE>>>(dev_x_, dev_y_, dev_z_, dev_l1norm_, dev_vtx_mapper_, radius_, num_vtx_,
                                         dev_num_neighbours_);
  CUDA_ERR_CHK(cudaPeekAtLastError());
  //std::cout << "Address of K: " << dev_num_neighbours_ << std::endl;
  CUDA_ERR_CHK(cudaMemcpy(&last_vtx_num_nbs, dev_num_neighbours_ + num_vtx_ - 1, sizeof(last_vtx_num_nbs), D2H));
  total_num_nbs_ += last_vtx_num_nbs;
}

void BFANET::Solver::calc_start_pos() {
  int last_vtx_start_pos = 0;

  const auto N = sizeof(dev_start_pos_[0]) * num_vtx_;

  // Do not free dev_start_pos_. It's required for the rest of algorithm.
  CUDA_ERR_CHK(cudaMalloc((void **)&dev_start_pos_, N));
  thrust::exclusive_scan(thrust::device, dev_num_neighbours_, dev_num_neighbours_ + num_vtx_, dev_start_pos_);
  CUDA_ERR_CHK(cudaMemcpy(&last_vtx_start_pos, dev_start_pos_ + num_vtx_ - 1, sizeof(int), D2H));
  total_num_nbs_ += last_vtx_start_pos;
}

void BFANET::Solver::append_neighbours() {
  const auto J = sizeof(dev_neighbours_[0]) * total_num_nbs_;

  CUDA_ERR_CHK(cudaMalloc((void **)&dev_neighbours_, J));

  k_append_neighbours<<<num_blocks_, BLOCK_SIZE>>>(dev_x_, dev_y_, dev_z_, dev_l1norm_, dev_vtx_mapper_,
                                                   dev_start_pos_, radius_, num_vtx_, dev_neighbours_);
  CUDA_ERR_CHK(cudaPeekAtLastError());

  // dev_x_ and dev_y_ are no longer used.
  CUDA_ERR_CHK(cudaFree(dev_x_));
  CUDA_ERR_CHK(cudaFree(dev_y_));
  CUDA_ERR_CHK(cudaFree(dev_z_));
  // graph has been fully constructed, hence free all the sorting related.
  CUDA_ERR_CHK(cudaFree(dev_l1norm_));
  CUDA_ERR_CHK(cudaFree(dev_vtx_mapper_));
  // dev_num_neighbours_, dev_start_pos_, dev_neighbours_ in GPU RAM.
}


void BFANET::Solver::avg_feat(float *feat_in_, float *feat_out_, int dim_num) {

    float *dev_feat_in_, *dev_feat_out_;

    CUDA_ERR_CHK(cudaMalloc((void **) &dev_feat_in_, sizeof(float) * num_vtx_*dim_num));
    CUDA_ERR_CHK(cudaMalloc((void **) &dev_feat_out_, sizeof(float) * num_vtx_*dim_num));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_sem_, sizeof(int) * num_vtx_));

    CUDA_ERR_CHK(cudaMemcpy(dev_feat_in_, feat_in_, sizeof(float) * num_vtx_*dim_num, cudaMemcpyHostToDevice));
    CUDA_ERR_CHK(cudaMemcpy(dev_sem_, sem_, sizeof(int) * num_vtx_, cudaMemcpyHostToDevice));

    // define block and threads
    dim3 block_n(DIVUP(num_vtx_, THREADS_PER_BLOCK));
    dim3 threads_n(THREADS_PER_BLOCK);
    search_mean_feat<<<block_n, threads_n>>>(dev_feat_in_, dev_feat_out_, dev_num_neighbours_, dev_start_pos_,
                                             dev_neighbours_, dev_sem_, num_vtx_, dim_num);

    CUDA_ERR_CHK(cudaMemcpy(feat_out_, dev_feat_out_, sizeof(int) * num_vtx_*dim_num, cudaMemcpyDeviceToHost));

    CUDA_ERR_CHK(cudaFree(dev_num_neighbours_));
    CUDA_ERR_CHK(cudaFree(dev_start_pos_));
    CUDA_ERR_CHK(cudaFree(dev_neighbours_));
    CUDA_ERR_CHK(cudaFree(dev_feat_in_));
    CUDA_ERR_CHK(cudaFree(dev_feat_out_));
    CUDA_ERR_CHK(cudaFree(dev_sem_));
}

int BFANET::Solver::identify_clusters(int cluster_accum,  vector<int>& clt_sem_vector) {

  int cluster = cluster_accum;
  for (int u = 0; u < num_vtx_; ++u) {
    if (cluster_ids[u] == -1) {
      bfs(u, cluster);
      clt_sem_vector.push_back(sem_[u]);
      ++cluster;
    }
  }
  CUDA_ERR_CHK(cudaFree(dev_num_neighbours_));
  CUDA_ERR_CHK(cudaFree(dev_start_pos_));
  CUDA_ERR_CHK(cudaFree(dev_neighbours_));

  return cluster;
}

void BFANET::Solver::bfs(const int u, const int cluster) {
    auto visited = new bool[num_vtx_]();
    auto frontier = new bool[num_vtx_]();
    int num_frontier = 1;
    frontier[u] = true;
    int sem_cur = sem_[u];
    const auto T = sizeof(visited[0]) * num_vtx_;
    const auto N = sizeof(x_[0]) * num_vtx_;

    bool *dev_visited, *dev_frontier;
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_visited, T));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_frontier, T));
    CUDA_ERR_CHK(cudaMalloc((void **)&dev_sem_, N));

    CUDA_ERR_CHK(cudaMemcpy(dev_visited, visited, T, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_frontier, frontier, T, H2D));
    CUDA_ERR_CHK(cudaMemcpy(dev_sem_, sem_, N, H2D));
    while (num_frontier > 0) {
        k_bfs<<<num_blocks_, BLOCK_SIZE>>>(dev_visited, dev_frontier, dev_num_neighbours_, dev_start_pos_,
                                           dev_neighbours_, dev_sem_, sem_cur, num_vtx_);
        CUDA_ERR_CHK(cudaPeekAtLastError());
        num_frontier = thrust::count(thrust::device, dev_frontier, dev_frontier + num_vtx_, true);
    }

    // we don't care about he content in dev_frontier now, hence no need to copy back.
    CUDA_ERR_CHK(cudaMemcpy(visited, dev_visited, T, D2H));
    CUDA_ERR_CHK(cudaFree(dev_visited));
    CUDA_ERR_CHK(cudaFree(dev_frontier));
    CUDA_ERR_CHK(cudaFree(dev_sem_));

    for (int n = 0; n < num_vtx_; ++n) {
        if (visited[n] && sem_[n] == sem_cur) {
            cluster_ids[n] = cluster;
        }
    }

    delete[] visited;
    delete[] frontier;
}