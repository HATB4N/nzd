#include "DenseLayer.h"
#include <random>
#include <ctime>

DenseLayer::DenseLayer(size_t row, size_t col, size_t batch, unsigned int threads) {
    _r = row;
    _c = col;
    _b = batch;
    _t = threads;

    _gemm = std::make_unique<Matrix>(_r, _c, _b, _t);
    _activator = std::make_unique<Activation>(_r, _c, _b);
}

void DenseLayer::forward() { // add args & delete test vector unit
    // test vector start
    std::vector<fp16> w(_r * _c);
    std::vector<fp16> x_t(_b * _c); // m2_col * m2_row transpose
    std::vector<fp16> b(_r);
    std::vector<fp32> r_t(_b * _r);

    for(size_t i = 0; i < w.size(); ++i) w[i] = static_cast<fp16>(random_float(-1.0f, 1.0f));
    for(size_t i = 0; i < x_t.size(); ++i) x_t[i] = static_cast<fp16>(random_float(-1.0f, 1.0f));
    for(size_t i = 0; i < b.size(); ++i) b[i] = static_cast<fp16>(random_float(-1.0f, 1.0f));
    // test vector end

    // unique_ptr로 계속 살아있게 한 다음에 thread pool 구현하기. IO 구조는 최대한 일관되게.
    _gemm->multiply(w, x_t, r_t);
    _gemm->add(r_t, b);
    _activator->l_relu(r_t); // template로 활성화 함수 선택 가능하게.
}

void DenseLayer::backward() { // 알고리즘 아직 ㅁㄹ

}
void DenseLayer::update() { // 알고리즘 아직 ㅁㄹ

}

fp16 DenseLayer::random_float(float fmin, float fmax) { // 처음 init용이니까 위치 바뀔 가능성 있음
    if (fmin > fmax) std::swap(fmin, fmax);

    thread_local std::mt19937 rng{ std::random_device{}() };

    std::uniform_real_distribution<float> dist(
        fmin, std::nextafter(fmax, std::numeric_limits<float>::infinity())
    );

    float sample = dist(rng);
    return static_cast<fp16>(sample);
}