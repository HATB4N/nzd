#include "Common/Matrix.h"
#include "Common/Struct.h"
#include <cmath>
#include <algorithm>
#include <thread>

Matrix::Matrix(unsigned int threads) {
    unsigned int MAX_T = std::thread::hardware_concurrency();
    if(!threads) threads = 1;
    this->_threads = std::min(threads, MAX_T);
}

void Matrix::set_threads(unsigned int threads) {
    unsigned int MAX_T = std::thread::hardware_concurrency();
    this->_threads = std::min(threads, MAX_T);
}