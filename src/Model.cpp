#include "Model.h"
#include "Activation.h"
#include "Initializers/HeInitializer.h"
#include <stdexcept>
#include "Activation.h" // enum매칭해?
#include "Common/Struct.h"

Model::Model() {}

int Model::init(uint64_t num_of_layers, // denselayer 기준
                uint64_t input_dim,
                uint64_t output_dim,
                uint64_t hidden_dim,
                uint64_t batch_size) {
    _nol = num_of_layers;
    _input_dim = input_dim;
    _output_dim =output_dim;
    _hidden_dim =hidden_dim;
    _batch_size =batch_size;
    unsigned int t = 14;
    auto he = std::make_shared<HeInitializer>(1); // seed injectin req

    uint64_t last_dim = _input_dim;
    // index = 0 | input layer | linear
    _layers.push_back(
        std::make_unique<DenseLayer>(ActFunc::LINEAR, 
                                     last_dim, 
                                     _hidden_dim, 
                                     he, 
                                     _layers.size()));

    // index = (1, _nol) | hidden layer | act
    for (uint64_t i = 0; i< _nol; i++) {
        last_dim = _hidden_dim;
        _layers.push_back(
        std::make_unique<DenseLayer>(ActFunc::RELU, 
                                     last_dim, 
                                     _hidden_dim, 
                                     he, 
                                     _layers.size()));
        }

    _layers.push_back(
        std::make_unique<DenseLayer>(ActFunc::LINEAR, // dz = p - y <-> last layer exception. act @ loss func
                                     last_dim, 
                                     _output_dim, 
                                     he, 
                                     _layers.size()));
    
    return 0; // add error accumulation (ret+=...)
}

Matrix_T<fp32> Model::forward_batch(const Matrix_T<fp16>& x) {
    Matrix_T<fp16> current_input = x; // cp
    Matrix_T<fp32> layer_output(_batch_size, _hidden_dim);

    for (uint64_t i = 0; i< _layers.size(); i++) {
        auto& layer = _layers[i];

        uint64_t current_output_dim =
            (i == _layers.size() - 1) ? _output_dim : _hidden_dim;

        if (layer_output.row() != _batch_size ||
            layer_output.col() != current_output_dim) {
            layer_output = Matrix_T<fp32>(_batch_size, current_output_dim);
        }

        layer->forward(current_input, layer_output); // 순수재미 Goat

        if (i < _layers.size() - 1) {
            if (current_input.row() != layer_output.row() ||
                current_input.col() != layer_output.col()) {
                current_input = Matrix_T<fp16>(layer_output.row(), layer_output.col());
            }
            const auto& out_data  = layer_output.data(View::NT);
            auto& next_data = current_input.data(View::NT);
            for (uint64_t j = 0; j < out_data.size(); ++j) {
                next_data[j] = static_cast<fp16>(out_data[j]);
            }
        }
    }
    return layer_output;
}

// WIP
Matrix_T<fp32> Model::backward_batch(const Matrix_T<fp32>& y) { // loss를 받음
    Matrix_T<fp32> current_grad = y; // 얘를 시작으로 상위 레이어 순회돌면서 역전파시키기
    Matrix_T<fp32> grad_output(_batch_size, _output_dim);
    for (uint64_t i = _layers.size()-1; i>= 0; i--) {
        auto& grad = _layers[i];

        uint64_t current_output_dim =
            (i == 0) ? _input_dim : (i == _layers.size()-1 ? _output_dim : _hidden_dim);

        if (grad_output.row() != _batch_size ||
            grad_output.col() != current_output_dim) {
            grad_output = Matrix_T<fp32>(_batch_size, current_output_dim);
        }

        grad->backward(current_grad, grad_output);

        if (i> 0) {
            if (current_grad.row() != grad_output.row() ||
                current_grad.col() != grad_output.col()) {
                current_grad = Matrix_T<fp32>(grad_output.row(), grad_output.col());
            }
            const auto& out_data  = grad_output.data(View::NT);
            auto& next_data = current_grad.data(View::NT);
            for (uint64_t j = 0; j < out_data.size(); ++j) {
                next_data[j] = static_cast<fp32>(out_data[j]);
            }
        }
    }
    return grad_output;
}