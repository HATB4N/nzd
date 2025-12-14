#include "Core/Model.h"
#include <stdexcept>
#include "Utils/Activation.h" // enum매칭해?
#include "Common/Struct.h"

Model::Model(uint64_t input_dim,
             uint64_t batch_size,
             InitType init,
             OptType opt) : _input_dim(input_dim),
                             _batch_size(batch_size),
                             _init(init),
                             _opt(opt) {
    _last_dim = _input_dim;
}
    
void Model::add(uint64_t dim, ActFunc act) {
    _layers.push_back(
        std::make_unique<DenseLayer>(act,
                                     _last_dim, 
                                     dim, 
                                     _init, 
                                     _opt,
                                     _layers.size()));
    _last_dim = dim;
}

Matrix_T<fp32> Model::forward_batch(const Matrix_T<fp32>& x) {
    const uint64_t current_batch_size = x.row();
    if (current_batch_size == 0) return Matrix_T<fp32>(0,0);

    Matrix_T<fp32> current_input = x;
    Matrix_T<fp32> layer_output(0, 0);

    for (uint64_t i = 0; i< _layers.size(); i++) {
        auto& layer = _layers[i];

        uint64_t current_output_dim = layer->get_out_dim();

        if (layer_output.row() != current_batch_size || layer_output.col() != current_output_dim) {
            layer_output = Matrix_T<fp32>(current_batch_size, current_output_dim);
        }

        layer->forward(current_input, layer_output);

        if (i < _layers.size() - 1) {
            if (current_input.row() != current_batch_size || current_input.col() != current_output_dim) {
                current_input = Matrix_T<fp32>(current_batch_size, current_output_dim);
            }
            std::swap(current_input, layer_output);
            // const auto& out_data  = layer_output.data(View::NT);
            // auto& next_data = current_input.data(View::NT);
            // #pragma omp parallel for
            // for (uint64_t j = 0; j < out_data.size(); ++j) {
            //     next_data[j] = static_cast<fp32>(out_data[j]);
            // }
        }
    }
    return layer_output;
}

Matrix_T<fp32> Model::backward_batch(const Matrix_T<fp32>& y) { // loss를 받음
    const uint64_t current_batch_size = y.row();
    if (current_batch_size == 0) return Matrix_T<fp32>(0,0);

    Matrix_T<fp32> current_grad = y;
    Matrix_T<fp32> grad_output(0,0);

    for (int i = _layers.size() - 1; i>= 0; i--) {
        auto& layer = _layers[i];

        // Determine the input dimension for the current layer to correctly size grad_output.
        // Based on the init() logic, layer 0 has `_input_dim` and all others have `_hidden_dim`.
        uint64_t input_dim_for_this_layer = layer->get_in_dim();

        if (grad_output.row() != current_batch_size || grad_output.col() != input_dim_for_this_layer) {
            grad_output = Matrix_T<fp32>(current_batch_size, input_dim_for_this_layer);
        }

        layer->backward(current_grad, grad_output);
        
        std::swap(current_grad, grad_output);
    }
    for (auto& layer : _layers) {
        layer->update();
    }
    return current_grad;
}