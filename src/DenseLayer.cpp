#include "DenseLayer.h"
#include "Struct.h"

DenseLayer::DenseLayer(unsigned int threads, void(*act)(Matrix_T<fp32>&)) {
        _t = threads;
        _act = act;
        _gemm = std::make_unique<Matrix>(_t);
    }

// R = σ(WX+b)
void DenseLayer::forward(Matrix_T<fp16> &w, Matrix_T<fp16> &x, Matrix_T<fp16> &b, Matrix_T<fp32> &r) {
    _gemm->multiply(w, x, r);
    _gemm->add(r, b);
    _act(r);
}

void DenseLayer::backward() { // 알고리즘 아직 ㅁㄹ

}
void DenseLayer::update() { // 알고리즘 아직 ㅁㄹ

}