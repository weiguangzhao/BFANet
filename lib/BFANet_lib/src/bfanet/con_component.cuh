#ifndef BFANET_CON_COMPONENT_CUH
#define BFANET_CON_COMPONENT_CUH

#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <vector>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))

// https://stackoverflow.com/a/14038590 to check the cuda status
#define CUDA_ERR_CHK(code) { cuda_err_chk((code), __FILE__, __LINE__); }

inline void cuda_err_chk(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "\tCUDA ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}


namespace BFANET {
int const BLOCK_SIZE = 512;

class Solver {
 public:
   Solver(float *, float *, float *, float *,  int *, int, float ,  int*, int * );

  //====Sort the input by the l1norm of each point.
  void sort_input_by_l1norm();

  //====calculate the number of neighbor
  void calc_num_neighbours();

  //====Prefix sum.
  void calc_start_pos();

  //====populate the actual neighbours for each vertex.
  void append_neighbours();

  //====cal the average feat
  void avg_feat(float*, float*, int);

  //==== Cluster HPs
  int identify_clusters(int,  std::vector<int>& );


 public:
  int *cluster_ids;

 private:
  cudaMemcpyKind D2H = cudaMemcpyDeviceToHost;
  cudaMemcpyKind H2D = cudaMemcpyHostToDevice;
  // query params
  int num_vtx_{};
  int total_num_nbs_{};
  // data structures
  float *x_{}, *y_{}, *z_{}, *l1norm_{};
  float radius_{};
  int *sem_{};
  // maps the sorted indices of each vertex to the original index.
  int *vtx_mapper_{};
  // gpu vars. Class members to avoid unnecessary copy.
  int num_blocks_{};
  float *dev_x_{}, *dev_y_{}, *dev_z_{}, *dev_l1norm_{};
  int *dev_sem_{};
  int *dev_vtx_mapper_{}, *dev_num_neighbours_{}, *dev_start_pos_{}, *dev_neighbours_{};
  int *dev_cluster_idx_{};


  void bfs(int u, int cluster);
};
}

#endif //BFANET_CON_COMPONENT_CUH