#include "Optimizers/SGD.h"

void SGD::step() { // test(예전 update하드코딩 복붙함)
    const float lr = 0.001f; // setter & private var로 지정
    auto& w_data = _W.data(View::NT);
    const auto& gw_data = _dW.data(View::NT);
    #pragma omp parallel for
    for (size_t i = 0; i < w_data.size(); ++i) {
        w_data[i] = static_cast<fp32>(static_cast<float>(w_data[i]) - lr * gw_data[i]);
    }
    auto& b_data = _b.data(View::NT);
    const auto& gb_data = _db.data(View::NT);
    #pragma omp parallel for
    for (size_t i = 0; i < b_data.size(); ++i) {
        b_data[i] = static_cast<fp32>(static_cast<float>(b_data[i]) - lr * gb_data[i]);
    }
}