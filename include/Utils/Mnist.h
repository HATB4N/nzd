#ifndef NZD_MNIST_H
#define NZD_MNIST_H

#include <fstream>
#include <string>
#include <iostream>
#include "Common/Types.h"
#include "Common/Struct.h"
#include <span>

static uint32_t read_uint32_t_big_endian(std::ifstream &fin) {
    uint32_t value = 0;
    for (uint32_t i = 0; i < sizeof(uint32_t); i++) {
        uint8_t byte;
        fin.read(reinterpret_cast<char *>(&byte), sizeof(byte));
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
    Mnist() = default;
    ~Mnist() = default;

    std::vector<uint8_t> all_labels;
    std::vector<uint8_t> all_images;

    int init(const std::string &f, const std::string &l) {
        // init labels
        std::ifstream fin_label(l, std::ios::binary);
        if (!fin_label.is_open()) { return -1; }
        uint32_t label_magicbyte = read_uint32_t_big_endian(fin_label);
        if (label_magicbyte != 0x00000801) return -1;
        uint32_t total_labels = read_uint32_t_big_endian(fin_label);
        all_labels.resize(total_labels);
        fin_label.read(reinterpret_cast<char *>(all_labels.data()), total_labels);
        fin_label.close();

        // init imgs
        std::ifstream fin_img(f, std::ios::binary);
        if (!fin_img.is_open()) { return -1; }
        uint32_t img_magicbyte = read_uint32_t_big_endian(fin_img);
        if (img_magicbyte != 0x00000803) return -1;
        this->_total_imgs = read_uint32_t_big_endian(fin_img);
        if (this->_total_imgs != total_labels) return -1;
        this->_rows = read_uint32_t_big_endian(fin_img);
        this->_cols = read_uint32_t_big_endian(fin_img);

        uint64_t image_data_size = static_cast<uint64_t>(_total_imgs) * _rows * _cols;
        all_images.resize(image_data_size);
        fin_img.read(reinterpret_cast<char *>(all_images.data()), image_data_size);
        fin_img.close();

        std::cout << "[MNIST]: init done, all data loaded into memory." << std::endl;
        return 0;
    }

    std::span<const uint8_t> get_image(uint64_t index) const {
        const size_t image_size = _rows * _cols;
        return std::span<const uint8_t>(&all_images[index * image_size], image_size);
    }

    uint8_t get_label(uint64_t index) const {
        return all_labels[index];
    }

    uint32_t get_total() const { return _total_imgs; }
    uint32_t get_rows() const { return _rows; }
    uint32_t get_cols() const { return _cols; }

private:
    uint32_t _total_imgs;
    uint32_t _rows;
    uint32_t _cols;
};

#endif // NZD_MNIST_H
