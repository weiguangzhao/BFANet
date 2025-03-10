#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "bfanet_lib.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("bfanet_cluster", &bfanet_cluster, "bfanet_cluster");
    m.def("sem_margin_det", &sem_margin_det, "sem_margin_det");
    m.def("avg_feat_radius", &avg_feat_radius, "avg_feat_radius");
}