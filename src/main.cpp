#include <iostream>
#include <stdfloat>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono> 
#include <memory>

#include "DenseLayer.h"
#include "Activation.h"
#include "Initializers/HeInitializer.h"

using fp16 = std::float16_t;
using fp32 = std::float32_t;
fp16 random_float(float, float);
void machine();

int main() {
    machine();
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);

    // 덴스레이어 테스트코드
    std::cout << "forward 1pass test" << std::endl;

    unsigned int threads = 14;
    size_t batch_size = 1024;
    size_t input_dim = 16384;
    size_t output_dim = 16384;

    Matrix_T<fp16> x(batch_size, input_dim);
    Matrix_T<fp32> r(batch_size, output_dim);

    auto& x_data = x.data(View::NT);
    for(size_t i = 0; i < x.size(); ++i) {
        x_data[i] = random_float(-1.0f, 1.0f);
    }

    auto he_initializer = std::make_unique<HeInitializer>();
    DenseLayer dense_layer(threads, &Act::relu, input_dim, output_dim, 0, std::move(he_initializer));

    auto start = std::chrono::high_resolution_clock::now();
    dense_layer.forward(x, r);
    auto finish = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = finish - start;
    std::cout << "Forward pass completed in: " << duration.count() << " sec" << std::endl;
    std::cout << "Result matrix r[0][0]: " << static_cast<float>(r.data(View::NT)[0]) << std::endl;
    std::cout << "Test finished." << std::endl;

    return 0;
}


fp16 random_float(float fmin, float fmax) {
    if (fmin > fmax) std::swap(fmin, fmax);

    thread_local std::mt19937 rng{ std::random_device{}() };

    std::uniform_real_distribution<float> dist(
        fmin, std::nextafter(fmax, std::numeric_limits<float>::infinity())
    );

    float sample = dist(rng);
    return static_cast<fp16>(sample);
}

void machine() {
    std::cout << "======================" << std::endl;
    std::cout << "from . import machine" << std::endl << std::endl;
    std::cout << "machine.learn()" << std::endl;
    std::cout << "======================" << std::endl << std::endl;
}