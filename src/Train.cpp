#include "Train.h"
#include "Activation.h"
#include <iostream>
#include <limits>
#include <algorithm>

enum class Job : uint8_t {
    FORWARD,
    BACKWARD
};
using JobFn = void(*)(Matrix_T<fp32>&);
void job_forward(Matrix_T<fp32> &r);
void job_backward(Matrix_T<fp32> &r);
constexpr JobFn JOB_TABLE[] = { // 나중에 ExecPlan으로
    &job_forward,
    &job_backward
};
inline JobFn target_job(Job j);

// 일단 mnist모듈이랑 합쳐둠. 나중에 load는 따로 설정하게
Train::Train(uint64_t epochs,
             uint64_t bpe, // 구조체로 hyper parms & config 설정해서 RW하게
             uint64_t input_dim, 
             uint64_t output_dim,
             uint64_t hidden_dim) : _epochs(epochs),
                                    _batches_per_epoch(bpe), 
                                    _model(std::make_unique<Model>()),
                                    _mnist(std::make_unique<Mnist>()),
                                    _input_dim(input_dim),
                                    _output_dim(output_dim),
                                    _hidden_dim(hidden_dim) {
    // 일단 임시로 달아두고, 나중에 data injection쪽 분리.
    std::string file = "dataset/train-images.idx3-ubyte";
    std::string label = "dataset/train-labels.idx1-ubyte";
    if (!_mnist->init(file, label)) exit(-1); // consume whole magic byte for sequential read
    this->_input_dim = static_cast<uint64_t>(_mnist->get_rows() * _mnist->get_cols());
    this->_total_data = static_cast<uint64_t>(_mnist->get_total());
    _model->init(10, _input_dim, _output_dim, _hidden_dim, _batches_per_epoch);
}

void Train::train_one_epoch() {
    uint64_t iter = (_total_data / _batches_per_epoch);

    for (uint64_t i = 0; i< iter; i++) {
        std::cout << "[Batch " << i+1 << " / " << iter << "] started.\n";

        // minibatch 데이터 읽어오기(mnist IO 구현 안해서 일단 랜덤. 파일 구조 보고 긁어올 수 있게)
        // 함수화해서 target batch를 load시키자.
        // 잔여 batch에 대한 alloc를 위해 기본값 말고, 재할당으로
        uint64_t _batch_size = _batches_per_epoch;
        Matrix_T<fp16> x(_batch_size, _input_dim);
        _batch_size = _mnist->get_batch<uint64_t>(_batches_per_epoch, x);

        // forward & 출력용 활성화(Softmax)
        Job flag = Job::FORWARD;
        JobFn output_layer_proc = target_job(flag);


        Matrix_T<fp32> logits = _model->forward_batch(x);
        output_layer_proc(logits); // softmax(logits)

        // TODO: 이제 역전파 구현
        // forward / backward 여부에 따라 output_layer_proc에 할당되는 함수가 달라짐.
        // 나중에는 이와 유사하게 단위 연산 단위로 분리해
        // 직렬화된 1차 Execution plan을 형성한 후, 위상 정렬 & 정적 스레드풀에 병렬 분배
        // 대충 컴파일러 로직 참고해서 만들면 될듯
    }
    _model->save_parms();
}

// 헬퍼
// Softmax 기준 actviation layer
void job_forward(Matrix_T<fp32> &r) {
    void (*_act)(Matrix_T<fp32> &) = resolve_act(ActFunc::SOFTMAX);
    _act(r);
}

void job_backward(Matrix_T<fp32> &r) {
    return;
}

inline JobFn target_job(Job j) {
    return JOB_TABLE[static_cast<size_t>(j)];
}