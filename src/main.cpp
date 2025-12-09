#include <iostream>
#include "Train.h"

void machine();

int main() {
    machine();
    Train train(1000, // max epochs
                256, // batch size
                28*28, // input dim
                10, // output dim
                1024); // hidden dim
    train.train_epoch();

    return 0;
}

void machine() {
    std::cout << "======================" << std::endl;
    std::cout << "from . import machine" << std::endl << std::endl;
    std::cout << "machine.learn()" << std::endl;
    std::cout << "======================" << std::endl << std::endl;
}
