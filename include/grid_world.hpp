#pragma once

#include <vector>
#include <utility>

class GridWorldEnv {
private:
    int width;
    int height;
    std::pair<int, int> agentPos;
    std::pair<int, int> goalPos;
    std::vector<std::pair<int, int>> obstacles;
    double stepReward;
    double goalReward;
    double obstacleReward;
    bool done;

public:
    GridWorldEnv(int w, int h);
    std::vector<double> reset();
    std::tuple<std::vector<double>, double, bool> step(int action);
    std::vector<double> getStateRepresentation() const;
    int getActionSpace() const;
    int getStateSize() const;
    void render() const;
};