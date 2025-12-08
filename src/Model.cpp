#include "Model.h"
#include "Activation.h"
#include "Initializers/HeInitializer.h"
#include <fstream> // ref save_parms / load_parms
#include <stdexcept>
#include "Activation.h" // enum매칭해?
#include "Common/Struct.h"

// for test
#include <random>
#include <limits>
#include <chrono>
#include <iostream>
#include <iomanip>

// for test
static fp16 random_float(float fmin, float fmax) {
    if (fmin > fmax) std::swap(fmin, fmax);
    thread_local std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<float> dist(
        fmin, std::nextafter(fmax, std::numeric_limits<float>::infinity())
    );
    float sample = dist(rng);
    return static_cast<fp16>(sample);
}

Model::Model(uint64_t num_of_layers) : _nol(num_of_layers) {}

void Model::init() {
    unsigned int t = 14;
    // 테스트
    // metadata는 배열에 저장해서 layer에 맞는 dim 읽어오게.
    // 지금은 그냥 임시 고정값.
    const uint64_t input_dim = 2048;
    const uint64_t output_dim = 128;
    const uint64_t hidden_dim = 8192;
    auto he = std::make_shared<HeInitializer>(1234);

    uint64_t last_dim = input_dim;

    // index = 0 | input layer | linear
    _layers.push_back(
        std::make_unique<DenseLayer>(ActFunc::LINEAR, 
                                     last_dim, 
                                     hidden_dim, 
                                     he, 
                                     _layers.size())); // 고유식별자임. 이 직렬화된 배열에서의 위치를 기반으로 나중에 그래프 구성. id -> datas (struct or class. idk)

    // index = (0, _nol) | hinnen layer | act
    last_dim = hidden_dim; // 임시
    for (uint64_t i = 1; i< _nol+1; i++) _layers.push_back(
        std::make_unique<DenseLayer>(ActFunc::RELU, 
                                     last_dim, 
                                     hidden_dim, 
                                     he, 
                                     _layers.size()));
    // // example of other usage                                 
    // _layers.push_back(
    //     std::make_unique<DenseLayer>(ActFunc::RELU, 
    //                                  last_dim, 
    //                                  hidden_dim, 
    //                                  nullptr, // without initialize
    //                                  _layers.size()));
    //     _layers[_layers.size()-1]->set_weight(_layers[_layers.size()-2]->get_weight()); // set data from another source

    // index = _nol+1 | output layer | softmax
    _layers.push_back(
        std::make_unique<DenseLayer>(ActFunc::SOFTMAX, 
                                     last_dim, 
                                     output_dim, 
                                     he, 
                                     _layers.size()));
}

// for test
void Model::test() {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "Model::test() forward propagation test started." << std::endl;

    // 전역으로 둬야 함. init에 맞게. 일단 테스트 코드
    const uint64_t input_dim = 2048;
    const uint64_t output_dim = 128;
    const uint64_t hidden_dim = 8192;
    const uint64_t batch_size = 1024;

    // 일단 임시로 랜덤
    Matrix_T<fp16> current_input(batch_size, input_dim);
    auto& x_data = current_input.data(View::NT);
    for(uint64_t i = 0; i < current_input.size(); ++i) {
        x_data[i] = random_float(-1.0f, 1.0f);
    }
    
    Matrix_T<fp32> layer_output(batch_size, hidden_dim); // init with first layer's output dimensions

    auto start = std::chrono::high_resolution_clock::now();

    // forward
    for(uint64_t i = 0; i < _layers.size(); i++) {
        auto& layer = _layers[i];
        
        // 현재 레이어의 출력 차원 결정
        uint64_t current_output_dim = (i == _layers.size() - 1) ? output_dim : hidden_dim;

        // 결과 행렬의 크기가 올바른지 확인
        if (layer_output.row() != batch_size || layer_output.col() != current_output_dim) {
            layer_output = Matrix_T<fp32>(batch_size, current_output_dim);
        }

        // unit forward
        layer->forward(current_input, layer_output);

        // 마지막 레이어가 아니면, 다음 레이어의 입력을 준비
        if (i < _layers.size() - 1) {
            // 다음 입력 행렬의 크기 조절
            if (current_input.row() != layer_output.row() || current_input.col() != layer_output.col()) {
                current_input = Matrix_T<fp16>(layer_output.row(), layer_output.col());
            }
            
            // 양자화 임시
            const auto& output_data = layer_output.data(View::NT);
            auto& next_input_data = current_input.data(View::NT);
            for(uint64_t j = 0; j < output_data.size(); j++) {
                next_input_data[j] = static_cast<fp16>(output_data[j]);
            }
        }
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = finish - start;
    
    std::cout << "Full model forward pass completed in: " << duration.count() << " sec" << std::endl;
    std::cout << "Final result matrix r[0][0]: " << static_cast<float>(layer_output.data(View::NT)[0]) << std::endl;
    std::cout << "Model::test() finished." << std::endl;
    
    save_parms();
}
