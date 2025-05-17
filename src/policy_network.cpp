#include "policy_network.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

std::vector<double> PolicyNetwork::softmax(const std::vector<double>& input) const {
    std::vector<double> result(input.size());
    double maxVal = *std::max_element(input.begin(), input.end());
    double sumExp = 0.0;

    for (size_t i = 0; i < input.size(); i++) {
        result[i] = std::exp(input[i] - maxVal);
        sumExp += result[i];
    }

    for (size_t i = 0; i < input.size(); i++) {
        result[i] /= sumExp;
    }

    return result;
}

double PolicyNetwork::relu(double x) const {
    return std::max(0.0, x);
}

PolicyNetwork::PolicyNetwork(const std::vector<int>& architecture, unsigned int seed) 
    : layerSizes(architecture), rng(seed) {
    weights.resize(architecture.size() - 1);
    biases.resize(architecture.size() - 1);
    std::uniform_real_distribution<double> dist(-0.1, 0.1);

    for (size_t i = 0; i < architecture.size() - 1; i++) {
        int inputSize = architecture[i];
        int outputSize = architecture[i + 1];
        weights[i].resize(outputSize);
        for (int neuron = 0; neuron < outputSize; neuron++) {
            weights[i][neuron].resize(inputSize);
            for (int input = 0; input < inputSize; input++) {
                double scale = std::sqrt(2.0 / inputSize);
                weights[i][neuron][input] = dist(rng) * scale;
            }
        }
        biases[i].resize(outputSize, 0.0);
    }
}

std::vector<double> PolicyNetwork::forward(const std::vector<double>& input) const {
    std::vector<double> currentActivation = input;

    for (size_t layer = 0; layer < weights.size() - 1; layer++) {
        std::vector<double> layerOutput(weights[layer].size());
        for (size_t neuron = 0; neuron < weights[layer].size(); neuron++) {
            double sum = biases[layer][neuron];
            for (size_t i = 0; i < currentActivation.size(); i++) {
                sum += weights[layer][neuron][i] * currentActivation[i];
            }
            layerOutput[neuron] = relu(sum);
        }
        currentActivation = layerOutput;
    }

    size_t outputLayer = weights.size() - 1;
    std::vector<double> logits(weights[outputLayer].size());
    for (size_t neuron = 0; neuron < weights[outputLayer].size(); neuron++) {
        double sum = biases[outputLayer][neuron];
        for (size_t i = 0; i < currentActivation.size(); i++) {
            sum += weights[outputLayer][neuron][i] * currentActivation[i];
        }
        logits[neuron] = sum;
    }

    return softmax(logits);
}

int PolicyNetwork::sampleAction(const std::vector<double>& state) const {
    std::vector<double> actionProbs = forward(state);
    std::discrete_distribution<int> actionDist(actionProbs.begin(), actionProbs.end());
    return actionDist(const_cast<std::mt19937&>(rng));
}

double PolicyNetwork::logProbability(const std::vector<double>& state, int action) const {
    std::vector<double> actionProbs = forward(state);
    return std::log(actionProbs[action]);
}

void PolicyNetwork::updateParameters(const std::vector<std::vector<std::vector<double>>>& weightGradients,
                                   const std::vector<std::vector<double>>& biasGradients,
                                   double learningRate) {
    for (size_t layer = 0; layer < weights.size(); layer++) {
        for (size_t neuron = 0; neuron < weights[layer].size(); neuron++) {
            for (size_t input = 0; input < weights[layer][neuron].size(); input++) {
                weights[layer][neuron][input] += learningRate * weightGradients[layer][neuron][input];
            }
            biases[layer][neuron] += learningRate * biasGradients[layer][neuron];
        }
    }
}

std::pair<std::vector<std::vector<std::vector<double>>>, std::vector<std::vector<double>>> 
PolicyNetwork::getParameters() const {
    return {weights, biases};
}