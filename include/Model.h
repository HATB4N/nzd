#ifndef MODEL_H
#define MODEL_H

#include "DenseLayer.h"
#include <string>
#include <vector>
#include <fstream>
#include <cstddef>

class Model {
public:
    Model(uint64_t num_of_layers);
    void init();
    void test(); // 테스트 코드 forward chain수행함.
    int save_parms(); // ret는 에러 확인용. throw 이용도 고려.
    int load_parms();

private:
    uint64_t _nol;
    // _nol: only count hidden layers (ignore static components)
    // 0: input layer
    // 1, 2, ..., _nol: hidden layer
    // _nol+1: output layer
    std::vector<std::unique_ptr<DenseLayer>> _layers;
    int save_unit_parms(uint64_t index, std::ofstream& _fout);
    int load_unit_parms(std::ifstream& _fin);
    std::string base_dir = "data/parms.bin"; // init시 초기화시켜

};

#endif