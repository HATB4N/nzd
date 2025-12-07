#include <iostream>
#include "Model.h"

void machine();

int main() {
    machine();
    // test code
    int i = 10;
    while(i--) {
        Model model(10);
        model.init();
        model.test();
    }
    
    return 0;
}

void machine() {
    std::cout << "======================" << std::endl;
    std::cout << "from . import machine" << std::endl << std::endl;
    std::cout << "machine.learn()" << std::endl;
    std::cout << "======================" << std::endl << std::endl;
}
