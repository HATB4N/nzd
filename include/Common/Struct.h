#ifndef NZD_STRUCT_H
#define NZD_STRUCT_H

#include "Common/Types.h"
#include <cstddef>
#include <vector>
#include <cstring>
#include <iostream>
#include <memory_resource>

enum class Ori { NT, T };
enum class View { NT, T };

template <typename T>
class Matrix_T {
public:
    Matrix_T(uint64_t row, uint64_t col, Ori init = Ori::NT, std::pmr::memory_resource* resource = std::pmr::get_default_resource())
    : _row(row), _col(col), _primary(init), _m(resource), _m_t(resource) {} // I just learn this cool init method

    // load parms에서 사용하는 생성자임.
    Matrix_T(uint64_t row, uint64_t col, const std::vector<T>& data_vec, Ori init = Ori::NT, std::pmr::memory_resource* resource = std::pmr::get_default_resource())
    : _m(data_vec.begin(), data_vec.end(), resource), get_NT(true), _row(row), _col(col), _primary(init), _m_t(resource) {
        if (data_vec.size() != row * col) {
            throw std::invalid_argument("data_vec size does not match row * col");
        }
    }

    uint64_t size() const { return _row * _col; }
    uint64_t row() const { return _row; }
    uint64_t col() const { return _col; }
    
    std::pmr::vector<T>& data(View view = View::NT) {
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

    const std::pmr::vector<T>& data(View view = View::NT) const {
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
    std::pmr::vector<T> _m;
    std::pmr::vector<T> _m_t;
    bool get_T = false, get_NT = false;
    bool expired_T = false, expired_NT = false;
    uint64_t _row, _col; // based on _m (not _m_t)
    Ori _primary;

    void _transpose_from_nt() { // _m_t를 채우는 것
        if (_m_t.size() != size()) _m_t.resize(size());
        for(uint64_t r = 0; r< _row; ++r) {
            for(uint64_t c = 0; c< _col; ++c) {
                _m_t[c*_row + r] = _m[r*_col + c];
            }
        }
        get_T = true;
        expired_T = false;
    }

    void _transpose_from_t() { // _m을 채우는 것
        if (_m.size() != size()) _m.resize(size());
        for(uint64_t r = 0; r< _row; ++r) {
            for(uint64_t c = 0; c< _col; ++c) {
                _m[r*_col + c] = _m_t[c*_row + r];
            }
        }
        get_NT = true;
        expired_NT = false;
    }
};

#endif