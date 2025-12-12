#include "Train.h"
#include "Activation.h"
#include <iostream>
#include <limits>
#include <algorithm>
#include "Common/Gemm.h"

// 일단 mnist모듈이랑 합쳐둠. 나중에 load는 따로 설정하게
Train::Train(uint64_t epochs,
             uint64_t bpe, // 구조체로 hyper parms & config 설정해서 RW하게
             uint64_t output_dim, // 얘도 읽어오게
             uint64_t hidden_dim) : _epochs(epochs),
                                    _batches_per_epoch(bpe), 
                                    _model(std::make_unique<Model>()),
                                    _output_dim(output_dim),
                                    _hidden_dim(hidden_dim) {}

int Train::init() {
    // -----LOAD MNIST BEGIN----- //
    std::string file = "dataset/train-images.idx3-ubyte";
    std::string label = "dataset/train-labels.idx1-ubyte";
    _mnist = std::make_unique<Mnist>(); // 범용 말고 일단 mnist로
    if (_mnist->init(file, label)) return -1; // consume whole magic byte for sequential read
    // -----LOAD MNIST END----- //
    this->_input_dim = static_cast<uint64_t>(_mnist->get_rows() * _mnist->get_cols());
    this->_total_data = static_cast<uint64_t>(_mnist->get_total());
    if (_model->init(2, _input_dim, _output_dim, _hidden_dim, _batches_per_epoch)) return -1;
    return 0;
}

void Train::train_one_epoch() {
    uint64_t iter = (_total_data / _batches_per_epoch);
    _current_idx = 0;

    for (uint64_t i = 0; i< iter; i++) {
        std::cout << "[Batch " << i+1 << " / " << iter << "] started.\n";

        // -----LOAD DATASET BEGIN----- //
        this->_batch_size = -1;
        Matrix_T<fp16> x = this->_load_dataset();
        Matrix_T<fp32> y = this->_get_label_batch_onehot();
        // -----LOAD DATASET END----- //

        // -----FORWARD 1 PASS BEGIN----- //
        Matrix_T<fp32> logits = _model->forward_batch(x); // forward
        // -----FORWARD 1 PASS END ----- //

        // -----BADKWARD 1 PASS BEGIN----- //
        gemm().sub<fp32, fp32>(logits, y); // get loss grad (OUTPUT == SOFTMAX)조건 등으로 나중에 일반화
        Matrix_T<fp32>dx = _model->backward_batch(logits);
        // -----BADKWARD 1 PASS END----- //
        
        _current_idx += _batch_size;
    }
    _model->save_parms(); // 수정
}

Matrix_T<fp32> Train::_get_label_batch_onehot() {
    assert(this->_batch_size> 0);
    Matrix_T<fp32> y(_batch_size, _output_dim); // row * col
    std::fill(y.data(View::NT).begin(), y.data(View::NT).end(), static_cast<fp32>(0.0f));
    fp32* ptr = y.data(View::NT).data();
    uint64_t cols = _output_dim;

    #pragma omp parallel for
    for (uint64_t i = 0; i < _batch_size; ++i) {
        // if (_current_idx + i >= _total_data) break; // for omp disabled

        uint8_t label = _mnist->all_labels[_current_idx + i];
        
        // One-Hot Encoding: (row=i, col=label) 위치를 1.0으로
        ptr[i * cols + label] = 1.0f;
    }
    return y;
}

Matrix_T<fp16> Train::_load_dataset() {
    this->_batch_size = _batches_per_epoch; // fix 먼저 몇개 남았나 가져와야함
    Matrix_T<fp16> x(_batch_size, _input_dim);
    _batch_size = _mnist->get_batch<uint64_t>(_batches_per_epoch, x);
    return x;
}