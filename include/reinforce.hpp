#pragma once

#include <memory>
#include <vector>
#include "policy_network.hpp"
#include "grid_world.hpp"

class REINFORCE {
private:
    std::unique_ptr<PolicyNetwork> policy;
    GridWorldEnv environment;
    double gamma;
    double learningRate;

    struct Transition {
        std::vector<double> state;
        int action;
        double reward;
        bool done;
    };

    std::vector<double> calculateReturns(const std::vector<double>& rewards) const;
    std::vector<double> normalizeReturns(const std::vector<double>& returns) const;

public:
    REINFORCE(double discountFactor = 0.99, double lr = 0.01, unsigned int seed = 42);
    std::pair<std::vector<Transition>, double> runEpisode();
    void updatePolicy(const std::vector<Transition>& trajectory);
    void train(int numEpisodes, int renderFrequency = 0);
    void renderEpisode();
};