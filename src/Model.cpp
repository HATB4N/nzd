#include "Model.h"
#include "Activation.h"
#include "Initializers/HeInitializer.h"
#include <fstream> // ref save_parms / load_parms
#include "Activation.h" // enum매칭해?
#include <stdexcept>
#include "Common/Struct.h"

// for test
#include <random>
#include <limits>
#include <chrono>
#include <iostream>
#include <iomanip>

static uint64_t read_uint64_t_big_endian(std::ifstream& fin);
void write_uint64_t_big_endian(std::vector<uint8_t>& vec, uint64_t target);

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
    const uint64_t input_dim = 768;
    const uint64_t output_dim = 10;
    const uint64_t hidden_dim = 1024;

    uint64_t last_dim = input_dim;

    // index = 0 | input layer | linear
    _layers.push_back(std::make_unique<DenseLayer>(t, &Act::relu, last_dim, hidden_dim, 0,
            std::make_unique<HeInitializer>()));;
    last_dim = hidden_dim;

    // index = (0, _nol) | hinnen layer | act
    for (uint64_t i = 1; i< _nol+1; i++) {
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
    const uint64_t input_dim = 768;
    const uint64_t output_dim = 10;
    const uint64_t hidden_dim = 1024;
    const uint64_t batch_size = 256;

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

int Model::save_parms() {
    int ret = 0;
    std::ofstream _fout;
    _fout.open(this->base_dir, std::ios::binary);
    if (!_fout.is_open()) {
        return -1;
    }
    // write magic byte(indentifier)
    std::vector<uint8_t> id = {'N', 'Z', 'D'};
    for(auto &c : id) _fout.write((char *)&c, sizeof(c));
    
    // _nol 기록
    std::vector<uint8_t> bytes;
    bytes.push_back(0x00);
    write_uint64_t_big_endian(bytes, _nol);
    bytes[0] = bytes.size()-1;
    for(auto &b : bytes) _fout.write((char *)&b, sizeof(b));

    for(uint64_t i = 0; i< _nol+2; i++) {
        ret += save_unit_parms(i, _fout);
    }
    // ret에 대한 예외 처리. 날려버릴지, 저장 없이 할지 정해야 함. 로드는 그냥 종료시키고
    _fout.close();
    return ret;
}

int Model::load_parms() {
    std::ifstream _fin(this->base_dir, std::ios::binary);
    if (!_fin.is_open()) {
        return -1;
    }
    // magic byte
    std::vector<uint8_t> test_id{3};
    _fin.read(reinterpret_cast<char*>(test_id.data()), 3);
    if (!_fin || std::string(test_id.begin(), test_id.end()) != "NZD") {
        return -1;
    }

    // _nol(레이어 개수) 읽기
    uint8_t nol_len;
    _fin.read(reinterpret_cast<char*>(&nol_len), sizeof(nol_len));
    
    std::vector<uint8_t> nol_bytes(nol_len);
    _fin.read(reinterpret_cast<char*>(nol_bytes.data()), nol_len);

    uint64_t file_nol = 0;
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
        return -1;
    }

    // 각 레이어의 파라미터를 순서대로 로드
    for(uint64_t i = 0; i< _nol + 2; i++) {
        if (load_unit_parms(_fin) != 0) {
            _fin.close();
            return -1;
        }
    }
    _fin.close();
    return 0; // 성공
}

int Model::save_unit_parms(uint64_t index, std::ofstream& _fout) {
    auto &target = _layers[index];
    auto &w_matrix = target->get_weight(); // including dimension information & actual datas(=parms)
    auto& w_data = w_matrix.data(View::NT);
    auto &b_matrix = target->get_bias(); // same as above
    auto& b_data = b_matrix.data(View::NT);
    
    // construct header
    std::vector<uint8_t> header;
    header.push_back(0x00); // for first byte, asuume that len_header <= 255byte (뒤에서 header[0] = header.size()-1로 길이 기록)
    write_uint64_t_big_endian(header, index);
    header.push_back(target->act_enum);
    write_uint64_t_big_endian(header, w_matrix.row());
    write_uint64_t_big_endian(header, w_matrix.col());
    write_uint64_t_big_endian(header, b_matrix.row());
    write_uint64_t_big_endian(header, b_matrix.col());
    header[0] = header.size()-1; // 첫 바이트(헤더 길이 제외)

    // write header
    for(auto &h : header) _fout.write((char *)&h, sizeof(h));

    // write parms
    _fout.write(reinterpret_cast<const char*>(w_data.data()), w_data.size() * sizeof(fp16));
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
    uint64_t index = read_uint64_t_big_endian(_fin);
    
    uint8_t act_enum_val;
    _fin.read(reinterpret_cast<char*>(&act_enum_val), sizeof(act_enum_val));

    uint64_t w_rows = read_uint64_t_big_endian(_fin);
    uint64_t w_cols = read_uint64_t_big_endian(_fin);
    uint64_t b_rows = read_uint64_t_big_endian(_fin);
    uint64_t b_cols = read_uint64_t_big_endian(_fin);

    if (index >= _layers.size()) return -1; // idx err

    auto& target_layer = _layers[index];

    // 파라미터 데이터 읽기
    uint64_t w_data_size = w_rows * w_cols;
    std::vector<fp16> w_data(w_data_size);
    _fin.read(reinterpret_cast<char*>(w_data.data()), w_data_size * sizeof(fp16));

    uint64_t b_data_size = b_rows * b_cols;
    std::vector<fp16> b_data(b_data_size);
    _fin.read(reinterpret_cast<char*>(b_data.data()), b_data_size * sizeof(fp16));

    if (_fin.fail()) return -1; // err

    try {
        target_layer->set_weight(Matrix_T<fp16>(w_rows, w_cols, w_data));
        target_layer->set_bias(Matrix_T<fp16>(b_rows, b_cols, b_data));
    } catch (const std::exception& e) {
        return -1; // err
    }

    return 0; // 성공
}

static uint64_t read_uint64_t_big_endian(std::ifstream& fin) {
    uint64_t value = 0;
    for (uint64_t i = 0; i < sizeof(uint64_t); ++i) {
        uint8_t byte;
        fin.read(reinterpret_cast<char*>(&byte), sizeof(byte));
        value = (value << 8) | byte;
    }
    return value;
}

void write_uint64_t_big_endian(std::vector<uint8_t>& vec, uint64_t target) {
    for (int i = sizeof(uint64_t)-1; i>= 0; i--) {
        uint64_t shifted_val = target >> (i * 8);
        uint8_t byte = (uint8_t)(shifted_val & 0xff);
        vec.push_back(byte);
    }    
}