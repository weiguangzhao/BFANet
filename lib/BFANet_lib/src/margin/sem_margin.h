//
// Created by weiguang zhao on 2024-01-02.
//

#ifndef DET_MARGIN_SEM_MARGIN_H
#define DET_MARGIN_SEM_MARGIN_H

void sem_margin_det(at::Tensor x_tensor, at::Tensor y_tensor, at::Tensor z_tensor, at::Tensor sem_tensor,
                    at::Tensor batch_index_tensor, at::Tensor margin_tensor, float radius, int batch_size);

#endif //DET_MARGIN_SEM_MARGIN_H
