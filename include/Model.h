#ifndef MODEL_H
#define MODEL_H

#include "DenseLayer.h"
#include <string>
#include <vector>
#include <fstream>

class Model {
public:
    Model(size_t num_of_layers);
    void init();
    void test(); // 테스트 코드 forward chain수행함.
    int save_parms(); // ret는 에러 확인용. throw 이용도 고려.
    int load_parms();

private:
    size_t _nol;
    std::vector<std::unique_ptr<DenseLayer>> _layers;
    int save_unit_parms(size_t index, std::ofstream& _fout);
    int load_unit_parms(std::ifstream& _fin);
    std::string base_dir = "data/parms.bin";

};

#endif