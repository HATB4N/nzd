#ifndef TRAIN_H
#define TRAIN_H

#include "Core/Model.h"
#include "Utils/Mnist.h"
#include <vector>

class Train {
public:
    Train(uint64_t epochs,
          uint64_t bpe,
          uint64_t nol,
          uint64_t output_dim,
          uint64_t hidden_dim);
    void train();
    void train_one_epoch();
    int init();
    void test();
private:
    std::unique_ptr<Model> _model;
    uint64_t _total_data;
    const uint64_t _epochs;
    const uint64_t _nol;
    const uint64_t _batch_size;
    uint64_t _input_dim;
    const uint64_t _output_dim;
    const uint64_t _hidden_dim;
    std::unique_ptr<Mnist> _mnist;
    std::vector<uint64_t> _data_indices;
    Matrix_T<fp32> _load_dataset();
    Matrix_T<fp32> _get_label_batch_onehot();
    uint64_t _current_idx = 0;
};

#endif // TRAIN_H