#include "Model.h"
#include "Activation.h"
#include "Initializers/HeInitializer.h"
#include <fstream> // ref save_parms / load_parms
#include <cstdint>
#include "Activation.h" // enum매칭해?
#include <stdexcept>
#include "Struct.h"

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

using fp16 = std::float16_t;
using fp32 = std::float32_t;

Model::Model(size_t num_of_layers) {
    _nol = num_of_layers;
}

void Model::init() {
    unsigned int t = 14;
    // 테스트
    // metadata는 배열에 저장해서 layer에 맞는 dim 읽어오게.
    // 지금은 그냥 임시 고정값.
    const size_t input_dim = 768;
    const size_t output_dim = 10;
    const size_t hidden_dim = 2048;

    size_t last_dim = input_dim;

    // index = 0 | input layer | linear
    _layers.push_back(std::make_unique<DenseLayer>(t, &Act::relu, last_dim, hidden_dim, 0,
            std::make_unique<HeInitializer>()));;
    last_dim = hidden_dim;

    // index = (0, _nol) | hinnen layer | act
    for (size_t i = 1; i< _nol+1; i++) {
        _layers.push_back(std::make_unique<DenseLayer>(t, &Act::relu, last_dim, hidden_dim, i,
            std::make_unique<HeInitializer>())); // Pass initializer
    }

    // index = _nol+1 | output layer | softmax
    _layers.push_back(std::make_unique<DenseLayer>(t, &Act::softmax, last_dim, output_dim, _nol+1,
        std::make_unique<HeInitializer>())); // Pass initializer
}

// for test
void Model::test() {
    std::cout.setf(std::ios::fixed);
    std::cout.precision(6);
    std::cout << "Model::test() forward propagation test started." << std::endl;

    // 전역으로 둬야 함. init에 맞게. 일단 테스트 코드
    const size_t input_dim = 768;
    const size_t output_dim = 10;
    const size_t hidden_dim = 2048;
    const size_t batch_size = 256;

    // 일단 임시로 랜덤
    Matrix_T<fp16> current_input(batch_size, input_dim);
    auto& x_data = current_input.data(View::NT);
    for(size_t i = 0; i < current_input.size(); ++i) {
        x_data[i] = random_float(-1.0f, 1.0f);
    }
    
    Matrix_T<fp32> layer_output(batch_size, hidden_dim); // init with first layer's output dimensions

    auto start = std::chrono::high_resolution_clock::now();

    // forward
    for(size_t i = 0; i < _layers.size(); ++i) {
        auto& layer = _layers[i];
        
        // 현재 레이어의 출력 차원 결정
        size_t current_output_dim = (i == _layers.size() - 1) ? output_dim : hidden_dim;

        // 결과 행렬의 크기가 올바른지 확인
        if (layer_output.row() != batch_size || layer_output.col() != current_output_dim) {
            layer_output = Matrix_T<fp32>(batch_size, current_output_dim);
        }

        // unit forward
        layer->forward(current_input, layer_output);

        // 마지막 레이어가 아니면, 다음 레이어의 입력을 준비
        if (i < _layers.size() - 1) {
            // 다음 입력 행렬의 크기 조절
            if (current_input.row() != batch_size || current_input.col() != current_output_dim) {
                current_input = Matrix_T<fp16>(batch_size, current_output_dim);
            }
            
            // 양자화 임시
            const auto& output_data = layer_output.data(View::NT);
            auto& next_input_data = current_input.data(View::NT);
            for(size_t j = 0; j < output_data.size(); ++j) {
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

// _nol: only count hidden layers (ignore static components)
// 0: input layer
// 1, 2, ..., _nol: hidden layer
// _nol+1: output layer


// save / load parms from storage
// 차원 정보가 보존된다면 transpose 여부에 따라 결정적으로 읽기 / 쓰기 방법의 규정이 가능함.
// 그러나, 어떠한 레이어에 속하는 어떠한 차원의 어떠한 파라미터인지를 구분하기 위한 매직 넘버가 필요함.
// 기본적으로 byte단위 RW가 읽어남. 즉, 자체적인 buffer에 저장한 다음에, consume하는 방식으로 RW를 진행한다.
// 특정한 bit patterns(byte-wise -> requires of buffer)을 만날 경우, 메타데이터를 참조하거나
// 자체적인 메타데이터 정보를 len_metadata와 함께 지정하여(byte 단위) metadata 영역에 대한 구분을 가능하게 한다
// fp16 / fp32에서 등장하지 않는 bit 패턴은 딱히 존재하지 않으며, bit 치환을 구현하는 건 비효율적.
// metadata영역부터 시작하여, 그냥 일반적인(e.g. iNode, http header) 프로토콜? 헤더와 같은 방식으로
// [metadata length - metadata - data length - data] 패턴의 반복으로 저장한다.
// getter만 구현하지 말고, denselayer 생성자에서 metadata에 맞는 크기의 matrix를 생성할 수 있게 setter 또한 구현한다.
// 일단 단위 함수로, 하나의 unit[metadata length - metadata - data length - data]에 대한 함수를 만들고, iterating

// OS level에서 file에 대한 lock을 거는 것 같음
// 함수 수준에서 save_parms / load_parms의 실행 여부를 확인하는 static variable busy르 규정하자.
// 전치의 cost자체는 그닥 크지 않으며, 얘는 많이 호출 될 함수도 아니니 그냥 특정 NT를 기준으로 저장하자.
// data length는 dimension으로 알 수 있으므로 생략.
// 최종적으로
// [metadata_length 1byte] [metadata] [data]
// [metadata] = [index] [Act_func_type] [w_row] [w_col] [b_row] [b_col]
// 나중에 앞에 magic byte추가

static size_t read_size_t_big_endian(std::ifstream& fin) {
    size_t value = 0;
    for (size_t i = 0; i < sizeof(size_t); ++i) {
        uint8_t byte;
        fin.read(reinterpret_cast<char*>(&byte), sizeof(byte));
        value = (value << 8) | byte;
    }
    return value;
}

int Model::save_parms() {
    int ret = 0;
    std::ofstream _fout;
    _fout.open(this->base_dir, std::ios::binary);
    if (!_fout.is_open()) {
        // 파일을 열 수 없을 경우 에러 처리
        return -1;
    }
    
    // _nol 기록
    std::vector<uint8_t> bytes;
    bytes.push_back(0x00);
    for (int i = sizeof(size_t)-1; i>= 0; i--) {
        size_t shifted_val = _nol >> (i * 8);
        uint8_t byte = (uint8_t)(shifted_val & 0xff);
        bytes.push_back(byte);
    }
    bytes[0] = bytes.size()-1;
    for(auto &b : bytes) _fout.write((char *)&b, sizeof(b));

    for(size_t i = 0; i<_nol+2; i++) {
        ret += save_unit_parms(i, _fout);
    }
    // ret에 대한 예외 처리. 날려버릴지, 저장 없이 할지 정해야 함. 로드는 그냥 종료시키고
    _fout.close();
    return ret;
}

int Model::load_parms() {
    std::ifstream _fin(this->base_dir, std::ios::binary);
    if (!_fin.is_open()) {
        // 파일을 열 수 없을 경우 에러 처리
        return -1;
    }

    // _nol(레이어 개수) 읽기
    uint8_t nol_len;
    _fin.read(reinterpret_cast<char*>(&nol_len), sizeof(nol_len));
    
    std::vector<uint8_t> nol_bytes(nol_len);
    _fin.read(reinterpret_cast<char*>(nol_bytes.data()), nol_len);

    size_t file_nol = 0;
    for(uint8_t byte : nol_bytes) {
        file_nol = (file_nol << 8) | byte;
    }

    // 모델의 레이어 개수와 파일에 저장된 레이어 개수가 일치하는지 확인
    if (file_nol != this->_nol) {
        _fin.close();
        return -1; // 아키텍처 불일치
    }

    // 모델이 초기화되었는지 확인
    if (_layers.size() != _nol + 2) {
        _fin.close();
        return -1; // 모델이 올바르게 초기화되지 않음
    }

    // 각 레이어의 파라미터를 순서대로 로드
    for(size_t i = 0; i < _nol + 2; i++) {
        if (load_unit_parms(_fin) != 0) {
            _fin.close();
            return -1; // 단위 에러
        }
    }
    _fin.close();
    return 0; // 성공
}

// assume that: fpointer is at the valid position(controlled by save_parms func.)
// read unit & accumulate ret
// 읽은 / 쓴 = consume한 양을 기록한 다음에, 호출측에 반환하거나(-1로 에러처리) 
// 순차 접근 가능하면 그냥 무시하고 호출측의 전역 f pointer를 기준으로 consume한다.
// 직렬화 대상은 파라미터 (내부 private vector)
// Tranpose 여부는 NT로 통합.
int Model::save_unit_parms(size_t index, std::ofstream& _fout) {
    auto &target = _layers[index];
    auto &w_matrix = target->get_weight(); // including dimension information & actual datas(=parms)
    auto& w_data = w_matrix.data(View::NT);
    auto &b_matrix = target->get_bias(); // same as above
    auto& b_data = b_matrix.data(View::NT);
    // 구성: [metadata_len_of_byte = 1byte][metadata][data]
    // data의 길이는 dim에 따라 결정적. 그냥 없어도 됨.
    // 우선 메타데이터를 저장하여 byte size를 구한다.
    // big endian 사용함.
    // metadata = [index][activie_func_type][w_dim_1][w_dim_2][b_dim_1][b_dim_2]
    
    // header
    std::vector<uint8_t> header;
    header.push_back(0x00); // for first byte, asuume that len_header < 255byte (뒤에서 header[0] = header.size()-1로 길이 기록)
    for (int j = sizeof(size_t)-1; j>= 0; j--) {
        size_t shifted_val = index >> (j * 8);
        uint8_t byte = (uint8_t)(shifted_val & 0xff);
        header.push_back(byte);
    }    
    header.push_back(target->act_enum);

    // 함수화 해야할듯...?
    // 아니면 그냥 쓰고, 쓴 길이 축적시키거나.
    for (int j = sizeof(size_t)-1; j>= 0; j--) {
        size_t shifted_val = w_matrix.row() >> (j * 8);
        uint8_t byte = (uint8_t)(shifted_val & 0xff);
        header.push_back(byte);
    }
    for (int j = sizeof(size_t)-1; j>= 0; j--) {
        size_t shifted_val = w_matrix.col() >> (j * 8);
        uint8_t byte = (uint8_t)(shifted_val & 0xff);
        header.push_back(byte);
    }

    for (int j = sizeof(size_t)-1; j>= 0; j--) {
        size_t shifted_val = b_matrix.row() >> (j * 8);
        uint8_t byte = (uint8_t)(shifted_val & 0xff);
        header.push_back(byte);
    }
    for (int j = sizeof(size_t)-1; j>= 0; j--) {
        size_t shifted_val = b_matrix.col() >> (j * 8);
        uint8_t byte = (uint8_t)(shifted_val & 0xff);
        header.push_back(byte);
    }
    header[0] = header.size()-1; // 첫 바이트(헤더 길이 제외)

    // 이제 벡터를 직렬화하여 메타데이터를 기록한다.
    // 이것보다, 이미 vector 내에 직렬화된 걸 한 번에 저장하는 방식으로 변경할 것.
    // 객체 자체에 대한 직렬화 ㅇㅇ
    for(auto &h : header) _fout.write((char *)&h, sizeof(h));

    // 데이터 영역, wieghts
    // for(auto &parm : w_matrix.data(View::NT)) _fout.write((char *)&parm, sizeof(parm));
    _fout.write(reinterpret_cast<const char*>(w_data.data()), w_data.size() * sizeof(fp16));
    // (a_11, a_12, ... a_1y, ... a_21, ..., a_xy) for M(x,y)
    // 데이터 영역, biases
    // for(auto &parm : b_matrix.data(View::NT)) _fout.write((char *)&parm, sizeof(parm));
    _fout.write(reinterpret_cast<const char*>(b_data.data()), b_data.size() * sizeof(fp16));
    return 0;
}

int Model::load_unit_parms(std::ifstream& _fin) {
    // 헤더 길이 읽기
    uint8_t header_len;
    _fin.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));

    // 종료조건
    if (_fin.eof()) {
        return 0; // 정상 종료 (ret< 0 -> errs)
    }

    // header
    size_t index = read_size_t_big_endian(_fin);
    
    uint8_t act_enum_val;
    _fin.read(reinterpret_cast<char*>(&act_enum_val), sizeof(act_enum_val));

    size_t w_rows = read_size_t_big_endian(_fin);
    size_t w_cols = read_size_t_big_endian(_fin);
    size_t b_rows = read_size_t_big_endian(_fin);
    size_t b_cols = read_size_t_big_endian(_fin);

    if (index >= _layers.size()) return -1; // idx err

    auto& target_layer = _layers[index];

    // 파라미터 데이터 읽기
    size_t w_data_size = w_rows * w_cols;
    std::vector<fp16> w_data(w_data_size);
    _fin.read(reinterpret_cast<char*>(w_data.data()), w_data_size * sizeof(fp16));

    size_t b_data_size = b_rows * b_cols;
    std::vector<fp16> b_data(b_data_size);
    _fin.read(reinterpret_cast<char*>(b_data.data()), b_data_size * sizeof(fp16));

    if (_fin.fail()) return -1; // err

    // 읽어온 데이터로 레이어의 파라미터 설정
    try {
        target_layer->set_weight(Matrix_T<fp16>(w_rows, w_cols, w_data));
        target_layer->set_bias(Matrix_T<fp16>(b_rows, b_cols, b_data));
    } catch (const std::exception& e) {
        return -1; // err
    }

    return 0; // 성공
}
