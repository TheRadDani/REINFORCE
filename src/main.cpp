#include "reinforce.hpp"
#include <iostream>

int main() {
    std::cout << "REINFORCE Algorithm Demonstration\n--------------------------------" << std::endl;
    REINFORCE agent(0.99, 0.01);
    agent.train(500, 100);
    std::cout << "\nFinal trained policy demonstration:" << std::endl;
    agent.renderEpisode();
    return 0;
}