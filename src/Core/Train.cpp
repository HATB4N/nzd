#include "Core/Train.h"
#include "Utils/Activation.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include <vector>
#include <numeric>
#include <random>
#include "Common/Gemm.h"

// 일단 mnist모듈이랑 합쳐둠. 나중에 load는 따로 설정하게
Train::Train(uint64_t epochs,
             uint64_t bpe, // 구조체로 hyper parms & config 설정해서 RW하게
             uint64_t nol,
             uint64_t output_dim, // 얘도 읽어오게
             uint64_t hidden_dim) : _epochs(epochs),
                                    _batches_per_epoch(bpe), 
                                    _nol(nol),
                                    _model(std::make_unique<Model>()),
                                    _output_dim(output_dim),
                                    _hidden_dim(hidden_dim) {}

int Train::init() {
    // -----LOAD MNIST BEGIN----- //
    std::string file = "dataset/train-images.idx3-ubyte";
    std::string label = "dataset/train-labels.idx1-ubyte";
    _mnist = std::make_unique<Mnist>(); // 범용 말고 일단 mnist로
    if (_mnist->init(file, label)) return -1;
    // -----LOAD MNIST END----- //
    this->_input_dim = static_cast<uint64_t>(_mnist->get_rows() * _mnist->get_cols());
    this->_total_data = static_cast<uint64_t>(_mnist->get_total());
    if (_model->init(_nol, _input_dim, _output_dim, _hidden_dim, _batches_per_epoch)) return -1;
    _data_indices.resize(_total_data);
    std::iota(_data_indices.begin(), _data_indices.end(), 0);
    return 0;
}

void Train::train() {
    std::random_device rd;
    std::mt19937 g(rd());

    for(uint64_t i = 0; i < _epochs; i++) {
        // Shuffle data indices at the beginning of each epoch
        std::shuffle(_data_indices.begin(), _data_indices.end(), g);
        
        std::cout << "[Epoches " << i + 1 << " / " << _epochs << "] started.\n";
        train_one_epoch();
    }
    // _model->save_parms();
    this->test();
}

void Train::test() {
    auto test_mnist = std::make_unique<Mnist>();
    std::string test_file = "dataset/t10k-images.idx3-ubyte";
    std::string test_label = "dataset/t10k-labels.idx1-ubyte";
    if (test_mnist->init(test_file, test_label)) {
        std::cerr << "Error while loading test dataset" << std::endl;
        return;
    }

    uint64_t test_total_data = test_mnist->get_total();
    uint64_t test_input_dim = test_mnist->get_rows() * test_mnist->get_cols();
    assert(test_input_dim == this->_input_dim);

    uint64_t test_current_idx = 0;
    uint64_t iter = (test_total_data + _batches_per_epoch - 1) / _batches_per_epoch;
    
    double total_loss = 0.0;
    uint64_t correct_predictions = 0;

    for (uint64_t i = 0; i <iter; i++) {
        uint64_t remaining_data = test_total_data > test_current_idx ? test_total_data - test_current_idx : 0;
        uint64_t current_batch_size = std::min((uint64_t)_batches_per_epoch, remaining_data);

        if (current_batch_size == 0) continue;

        // labels
        Matrix_T<fp32> y(current_batch_size, _output_dim);
        auto& y_data = y.data(View::NT);
        std::fill(y_data.begin(), y_data.end(), 0.0f);
        std::vector<uint8_t> true_labels;
        true_labels.resize(current_batch_size);

        for (uint64_t j = 0; j < current_batch_size; ++j) {
            uint8_t label = test_mnist->all_labels[test_current_idx + j];
            y_data[j * _output_dim + label] = 1.0f;
            true_labels[j] = label;
        }

        // imgs
        Matrix_T<fp32> x(current_batch_size, test_input_dim);
        auto& x_data = x.data(View::NT);
        for (uint64_t j = 0; j < current_batch_size; ++j) {
            auto image_span = test_mnist->get_image(test_current_idx + j);
            fp32* dst_ptr = &x_data[j * test_input_dim];
            for (size_t p = 0; p < test_input_dim; ++p) {
                dst_ptr[p] = static_cast<fp32>(image_span[p] / 255.0f);
            }
        }

        // fw
        Matrix_T<fp32> logits = _model->forward_batch(x);

        // score
        const auto& pred_data = logits.data(View::NT);
        const float epsilon = 1e-7f;

        for (size_t j = 0; j < current_batch_size; ++j) {
            // loss
            if (true_labels[j] < _output_dim) {
                 total_loss += -std::log(pred_data[j * _output_dim + true_labels[j]] + epsilon);
            }

            // acc
            auto row_start = pred_data.begin() + j * _output_dim;
            auto row_end = row_start + _output_dim;
            uint64_t predicted_label = std::distance(row_start, std::max_element(row_start, row_end));
            if (predicted_label == true_labels[j]) {
                correct_predictions++;
            }
        }
        
        test_current_idx += current_batch_size;
    }

    double avg_loss = total_loss / test_total_data;
    double accuracy = static_cast<double>(correct_predictions) / test_total_data;

    std::cout << "\n----- Test Results -----\n";
    std::cout << "Average Loss: " << avg_loss << std::endl;
    std::cout << "Accuracy: " << accuracy * 100.0 << " %" << std::endl;
    std::cout << "------------------------\n";
}

void Train::train_one_epoch() {
    uint64_t iter = (_total_data + _batches_per_epoch - 1) / _batches_per_epoch;
    _current_idx = 0;

    for (uint64_t i = 0; i < iter; i++) {
        // -----LOAD DATASET BEGIN----- //
        Matrix_T<fp32> y = this->_get_label_batch_onehot();
        if (_batch_size == 0) continue; // blank cond.
        Matrix_T<fp32> x = this->_load_dataset();
        // -----LOAD DATASET END----- //

        // -----FORWARD 1 PASS BEGIN----- //
        Matrix_T<fp32> logits = _model->forward_batch(x);
        // -----FORWARD 1 PASS END ----- //

        // dbg
        if (i == 0) {
            double total_loss = 0.0;
            const size_t BATCH_SIZE = logits.row();
            const size_t OUT_DIM = logits.col();
            
            const auto& pred_data = logits.data(View::NT);
            const auto& label_data = y.data(View::NT);
            const float epsilon = 1e-7f;

            for (size_t k = 0; k < BATCH_SIZE * OUT_DIM; ++k) {
                if (label_data[k] > 0.9f) {
                    total_loss += -std::log(pred_data[k] + epsilon);
                }
            }
            
            std::cout << "Batch Loss: " << (total_loss / BATCH_SIZE) << std::endl;
        }

        // -----BADKWARD 1 PASS BEGIN----- //
        gemm().sub<fp32, fp32>(logits, y); // ome-hot encoding
        Matrix_T<fp32> dx = _model->backward_batch(logits);
        // -----BADKWARD 1 PASS END----- //
        
        _current_idx += _batch_size;
    }
}

Matrix_T<fp32> Train::_get_label_batch_onehot() {
    uint64_t remaining_data = _total_data > _current_idx ? _total_data - _current_idx : 0;
    _batch_size = std::min((uint64_t)_batches_per_epoch, remaining_data);

    if (_batch_size == 0) {
        return Matrix_T<fp32>(0, 0);
    }

    Matrix_T<fp32> y(_batch_size, _output_dim);
    auto& y_data = y.data(View::NT);
    std::fill(y_data.begin(), y_data.end(), 0.0f);

    #pragma omp parallel for
    for (uint64_t i = 0; i < _batch_size; ++i) {
        uint64_t data_index = _data_indices[_current_idx + i];
        uint8_t label = _mnist->get_label(data_index);
        y_data[i * _output_dim + label] = 1.0f;
    }
    return y;
}

Matrix_T<fp32> Train::_load_dataset() {
    if (_batch_size == 0) {
        return Matrix_T<fp32>(0, 0);
    }

    Matrix_T<fp32> x(_batch_size, _input_dim);
    auto& x_data = x.data(View::NT);

    #pragma omp parallel for
    for (uint64_t i = 0; i < _batch_size; ++i) {
        uint64_t data_index = _data_indices[_current_idx + i];
        auto image_span = _mnist->get_image(data_index);
        
        fp32* dst_ptr = &x_data[i * _input_dim];
        for (size_t p = 0; p < _input_dim; ++p) {
            dst_ptr[p] = static_cast<fp32>(image_span[p] / 255.0f);
        }
    }
    return x;
}