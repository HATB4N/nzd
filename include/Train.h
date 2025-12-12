#ifndef TRAIN_H
#define TRAIN_H

#include "Model.h"
#include "Utils/Mnist.h"

class Train {
public:
    Train(uint64_t epochs,
          uint64_t bpe,
          uint64_t output_dim,
          uint64_t hidden_dim);
    void train_one_epoch();
    int init();
private:
    std::unique_ptr<Model> _model;
    uint64_t _total_data;
    const uint64_t _epochs;
    uint64_t _batches_per_epoch;
    uint64_t _batch_size;
    uint64_t _input_dim;
    const uint64_t _output_dim;
    const uint64_t _hidden_dim;
    std::unique_ptr<Mnist> _mnist;
    Matrix_T<fp16> _load_dataset();
    Matrix_T<fp32> _get_label_batch_onehot();
    uint64_t _current_idx = 0;
};

#endif // TRAIN_H