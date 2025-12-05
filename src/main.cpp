#include <iostream>
#include <stdfloat>
#include <iomanip>
#include <vector>
#include <random>
#include <chrono> 
#include <memory>

#include "DenseLayer.h"
#include "Activation.h"
#include "Initializers/HeInitializer.h"
#include "Model.h"

using fp16 = std::float16_t;
using fp32 = std::float32_t;
fp16 random_float(float, float);
void machine();

int main() {
    machine();
    // test code
    Model model(16);
    model.init();
    model.test();

    return 0;
}

void machine() {
    std::cout << "======================" << std::endl;
    std::cout << "from . import machine" << std::endl << std::endl;
    std::cout << "machine.learn()" << std::endl;
    std::cout << "======================" << std::endl << std::endl;
}
