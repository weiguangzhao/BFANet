#ifndef BFANET_FUNCTIONS_CUH
#define BFANET_FUNCTIONS_CUH

__global__ void k_num_nbs(float const *const x, float const *const y, float const *const z,
                          float const *const l1norm, int const *const vtx_mapper, float radius_,
                          const int num_vtx, int *const num_nbs);

__global__ void k_append_neighbours(float const *const x, float const *const y, float const *const z,
                                    float const *const l1norm, int const *const vtx_mapper, int const *const start_pos,
                                    float radius_, const int num_vtx, int *const neighbours);

__global__ void k_bfs(bool *const visited, bool *const frontier,
                      int const *const num_nbs,
                      int const *const start_pos,
                      int const *const neighbours,
                      int *dev_sem_,
                      int sem_cur,
                      int num_vtx);


__device__ inline float square_dist(const float x1, const float y1, const float z1,
                                    const float x2, const float y2, const float z2);

__global__ void get_cc_id(bool *const dev_visited, int *dev_cluster_idx_, int *dev_sem_, int cluster, int sem_cur,
                          int num_vtx);

__global__ void search_mean_feat(float *dev_feat_in, float *dev_feat_out,
                                 int const *const num_nbs,
                                 int const *const start_pos,
                                 int const *const neighbours,
                                 int *dev_sem_, int num_vtx_, int dim_num);


#endif //BFANET_FUNCTIONS_CUH
