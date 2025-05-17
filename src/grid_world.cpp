#include "grid_world.hpp"
#include <iostream>
#include <algorithm>

GridWorldEnv::GridWorldEnv(int w, int h) 
    : width(w), height(h), stepReward(-0.1), goalReward(1.0), obstacleReward(-1.0), done(false) {
    goalPos = {width - 1, height - 1};
    obstacles = {{1, 1}, {1, 2}, {2, 1}, {width - 2, height - 2}};
    reset();
}

std::vector<double> GridWorldEnv::reset() {
    agentPos = {0, 0};
    done = false;
    return getStateRepresentation();
}

std::tuple<std::vector<double>, double, bool> GridWorldEnv::step(int action) {
    auto oldPos = agentPos;
    switch (action) {
        case 0: agentPos.second = std::max(0, agentPos.second - 1); break;
        case 1: agentPos.first = std::min(width - 1, agentPos.first + 1); break;
        case 2: agentPos.second = std::min(height - 1, agentPos.second + 1); break;
        case 3: agentPos.first = std::max(0, agentPos.first - 1); break;
    }

    bool hitObstacle = false;
    for (const auto& obs : obstacles) {
        if (agentPos == obs) {
            hitObstacle = true;
            agentPos = oldPos;
            break;
        }
    }

    double reward = stepReward;
    if (hitObstacle) reward = obstacleReward;
    if (agentPos == goalPos) {
        reward = goalReward;
        done = true;
    }

    return {getStateRepresentation(), reward, done};
}

std::vector<double> GridWorldEnv::getStateRepresentation() const {
    return {
        static_cast<double>(agentPos.first) / (width - 1),
        static_cast<double>(agentPos.second) / (height - 1)
    };
}

int GridWorldEnv::getActionSpace() const {
    return 4;
}

int GridWorldEnv::getStateSize() const {
    return 2;
}

void GridWorldEnv::render() const {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (std::make_pair(x, y) == agentPos) std::cout << "A ";
            else if (std::make_pair(x, y) == goalPos) std::cout << "G ";
            else {
                bool isObstacle = std::find(obstacles.begin(), obstacles.end(), std::make_pair(x, y)) != obstacles.end();
                std::cout << (isObstacle ? "# " : ". ");
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}