#pragma once

#include <vector>
#include <random>

class PolicyNetwork {
private:
    std::vector<int> layerSizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::mt19937 rng;

    std::vector<double> softmax(const std::vector<double>& input) const;
    double relu(double x) const;

public:
    PolicyNetwork(const std::vector<int>& architecture, unsigned int seed = 42);
    std::vector<double> forward(const std::vector<double>& input) const;
    int sampleAction(const std::vector<double>& state) const;
    double logProbability(const std::vector<double>& state, int action) const;
    void updateParameters(const std::vector<std::vector<std::vector<double>>>& weightGradients,
                          const std::vector<std::vector<double>>& biasGradients,
                          double learningRate);
    std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> getParameters() const;
};