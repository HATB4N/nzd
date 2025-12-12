#include <iostream>
#include "Train.h"

void machine();

int main() {
    machine();
    Train train(50,
                128, // batch size
                10, // output dim
                128); // hidden dim
    if(train.init()) exit(-1);
    train.train();

    return 0;
}

void machine() {
    std::cout << "======================" << std::endl;
    std::cout << "from . import machine" << std::endl << std::endl;
    std::cout << "machine.learn()" << std::endl;
    std::cout << "======================" << std::endl << std::endl;
}
