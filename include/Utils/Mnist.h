#ifndef NZD_MNIST_H
#define NZD_MNIST_H

#include <fstream>
#include <string>
#include <iostream>
#include "Common/Types.h"
#include "Common/Struct.h"

static uint32_t read_uint32_t_big_endian(std::ifstream& fin) {
    uint32_t value = 0;
    for (uint32_t i = 0; i< sizeof(uint32_t); i++) {
        uint8_t byte;
        fin.read(reinterpret_cast<char*>(&byte), sizeof(byte));
        value = (value << 8) | byte;
    }
    return value;
}

// static uint32_t read_uint32_t_little_endian(std::ifstream& fin) {
//     uint32_t value = 0;
//     for (uint32_t i = 0; i < sizeof(uint32_t); i++) {
//         uint8_t byte;
//         fin.read(reinterpret_cast<char*>(&byte), sizeof(byte));
//         value |= (static_cast<uint32_t>(byte) << (i * 8));
//     }
//     return value;
// }

class Mnist {
public:
    Mnist() {}
    ~Mnist() {
        this->_fin_img.close();
        this->_fin_label.close();
    }
    std::vector<uint8_t> all_labels;
    int init(std::string f, std::string l) {
        this->_path_img = f;
        this->_path_label = l;
        this->_fin_img = std::ifstream(_path_img, std::ios::binary);
        this->_fin_label = std::ifstream(_path_label, std::ios::binary);
        if (!_fin_img.is_open() || !_fin_label.is_open()) {
            std::cout << "[MNIST]: cannot open the file." << std::endl;
            return -1;
        } std::cout << "[MNIST]: the file opened successfully." << std::endl;   

        uint32_t img_magicbyte = read_uint32_t_big_endian(_fin_img);
        if (!(img_magicbyte == 0x00000803)) return -1;
        std::cout << "[MNIST]: read valid img magicbyte" << std::endl;

        this->_total_imgs = read_uint32_t_big_endian(_fin_img);
        std::cout << "[MNIST]: total imgs: " << _total_imgs << std::endl;

        this->_rows = read_uint32_t_big_endian(_fin_img);
        std::cout << "[MNIST]: rows: " << _rows << std::endl;

        this->_cols = read_uint32_t_big_endian(_fin_img);
        std::cout << "[MNIST]: cols: " << _cols << std::endl;
        
        uint32_t label_magicbyte = read_uint32_t_big_endian(_fin_label);
        if (!(label_magicbyte == 0x00000801)) return -1;

        std::cout << "[MNIST]: read valid label magicbyte" << std::endl;
        if (!(this->_total_imgs == read_uint32_t_big_endian(_fin_label))) return -1;
        this->all_labels.resize(_total_imgs);
        this->_fin_label.read(reinterpret_cast<char*>(all_labels.data()), sizeof(uint8_t)*_total_imgs);
        std::cout << "[MNIST]: init done" << std::endl;
        return 0;
    }

    template <typename T>
    T get_batch(T exp_b, Matrix_T<fp16> &x) {
        uint64_t num_elements = exp_b * this->_rows * this->_cols;
        
        std::vector<uint8_t> temp_buf(num_elements);
        _fin_img.read(reinterpret_cast<char*>(temp_buf.data()), num_elements * sizeof(uint8_t));
        std::streamsize bytes_read = _fin_img.gcount();
        uint64_t actual_elements = static_cast<uint64_t>(bytes_read);
        fp16* dst_ptr = x.data(View::NT).data();
        #pragma omp parallel for
        for (uint64_t i = 0; i < actual_elements; ++i) {
            // f: N[0, 255] -> R[0.0, 1.0] @ fp16
            dst_ptr[i] = static_cast<fp16>(temp_buf[i] / 255.0f);
        }
        return static_cast<T>(actual_elements); 
    }

    
        
    uint32_t get_total() { return _total_imgs; }
    uint32_t get_rows() { return _rows; }
    uint32_t get_cols() { return _cols; }

private:
    std::string _path_img;
    std::string _path_label;
    std::ifstream _fin_img;
    std::ifstream _fin_label;
    uint32_t _total_imgs;
    uint32_t _rows;
    uint32_t _cols;
};

#endif // NZD_MNIST_H