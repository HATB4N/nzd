#include "Matrix.h"
#include <cmath>
#include <algorithm>
#include <thread>

Matrix::Matrix(unsigned int threads) {
    unsigned int MAX_T = std::thread::hardware_concurrency();
    if(!threads) threads = 1;
    this->_threads = std::min(threads, MAX_T);
}

Matrix::~Matrix() {
    // 나중에 쓰레드풀 정적으로 두고 여기서 정리시키기
}

void Matrix::set_threads(unsigned int threads) {
    unsigned int MAX_T = std::thread::hardware_concurrency();
    this->_threads = std::min(threads, MAX_T);
}