#include <iostream>
#include "Core/Train.h"

void machine();

int main() {
    machine();
    Train train(30, // epoches
                128, // batch size
                3, // hidden layer num
                10, // output dim
                64); // hidden dim
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
