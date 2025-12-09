#include "LossFunc/CrossEntropy.h"
#include <cmath>
#include <cassert>
#include <numeric>

fp32 CrossEntropy::calculate(const Matrix_T<fp32>& y_pred, const Matrix_T<fp32>& y_true) {
    assert(y_pred.row() == y_true.row() && y_pred.col() == y_true.col());

    uint64_t batch_size = y_pred.row();
    uint64_t num_classes = y_pred.col();
    const auto& y_pred_data = y_pred.data(View::NT);
    const auto& y_true_data = y_true.data(View::NT);

    fp32 total_loss = 0.0f;
    const fp32 epsilon = 1e-9f; // -inf 방지

    for (uint64_t i = 0; i < batch_size; ++i) {
        for (uint64_t j = 0; j < num_classes; ++j) {
            // y_true가 one-hot 벡터라고 가정하고, 1.0f인 위치의 예측 확률에 대해서만 log loss를 계산
            if (y_true_data[i * num_classes + j] == 1.0f) {
                total_loss += -std::log(y_pred_data[i * num_classes + j] + epsilon);
                break; // one-hot이므로 해당 행(row)의 계산을 중단
            }
        }
    }

    return total_loss / static_cast<fp32>(batch_size); // 배치의 평균 손실을 반환
}

void CrossEntropy::backward(const Matrix_T<fp32>& y_pred, const Matrix_T<fp32>& y_true, Matrix_T<fp32>& gradient) {
    assert(y_pred.row() == y_true.row() && y_pred.col() == y_true.col());
    assert(gradient.row() == y_pred.row() && gradient.col() == y_pred.col());

    size_t total_size = y_pred.size();
    const auto& y_pred_data = y_pred.data(View::NT);
    const auto& y_true_data = y_true.data(View::NT);
    auto& grad_data = gradient.data(View::NT);

    // 역전파의 시작 그래디언트는 (예측값 - 실제값)
    for (uint64_t i = 0; i < total_size; ++i) {
        grad_data[i] = y_pred_data[i] - y_true_data[i];
    }
}
