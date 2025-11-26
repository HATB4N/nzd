#ifndef STRUCT_T
#define STRUCT_T

#include <stdfloat>
#include <cstddef>
#include <vector>
#include <cstring>
#include <iostream>

using fp16 = std::float16_t;
using fp32 = std::float32_t;

enum class Ori { NT, T };
enum class View { NT, T };

template <typename T>
class Matrix_T {
public:
    Matrix_T(size_t row, size_t col, Ori init = Ori::NT)
    : _row(row), _col(col), _primary(init) {} // I just learn this cool init method

    size_t size() const { return _row * _col; }
    size_t row() const { return _row; }
    size_t col() const { return _col; }
    
    std::vector<T>& data(View view = View::NT) {
        if(view == View::NT) {
            if(_m.empty()) {
                if(get_T && !get_NT && !expired_T) {
                    _transpose_from_t();
                } else {
                    _m.resize(size());
                    get_NT = true;
                } 
            }
            expired_T = true;
            return _m;
        } else {
            if(_m_t.empty()) {
                if (get_NT && !get_T && !expired_NT) {
                    _transpose_from_nt();
                } else {
                    _m_t.resize(size());
                    get_T = true;
                }
            }
        }
        expired_NT = true;
        return _m_t;
    }

    const std::vector<T>& data(View view = View::NT) const {
        if (view == View::NT) {
            if (_m.empty()) {
                if (get_T && !expired_T) {
                    const_cast<Matrix_T*>(this)->_transpose_from_t();
                } else {
                    const_cast<Matrix_T*>(this)->_m.resize(size());
                    const_cast<Matrix_T*>(this)->get_NT = true;
                }
            } return _m;
        } else {
            if (_m_t.empty()) {
                if (get_NT && !expired_NT) {
                    const_cast<Matrix_T*>(this)->_transpose_from_nt();
                } else {
                    const_cast<Matrix_T*>(this)->_m_t.resize(size()); 
                    const_cast<Matrix_T*>(this)->get_T = true;
                }
            } return _m_t;
        }
    }

private:
    std::vector<T> _m;
    std::vector<T> _m_t;
    bool get_T = false, get_NT = false;
    bool expired_T = false, expired_NT = false;
    size_t _row, _col; // based on _m (not _m_t)
    Ori _primary;

    void _transpose_from_nt() { // _m_t를 채우는 것
        std::cout << "tfnt" << std::endl;
        if (_m_t.size() != size()) _m_t.resize(size());
        for(size_t r = 0; r< _row; ++r) {
            for(size_t c = 0; c< _col; ++c) {
                _m_t[c*_row + r] = _m[r*_col + c];
            }
        }
        get_T = true;
        expired_T = false;
    }

    void _transpose_from_t() { // _m을 채우는 것
        std::cout << "tft" << std::endl;
        if (_m.size() != size()) _m.resize(size());
        for(size_t r = 0; r< _row; ++r) {
            for(size_t c = 0; c< _col; ++c) {
                _m[r*_col + c] = _m_t[c*_row + r];
            }
        }
        get_NT = true;
        expired_NT = false;
    }
};

#endif