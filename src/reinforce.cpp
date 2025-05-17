#include "reinforce.hpp"
#include <iostream>
#include <numeric>
#include <cmath>

std::vector<double> REINFORCE::calculateReturns(const std::vector<double>& rewards) const {
    std::vector<double> returns(rewards.size());
    double cumulative = 0.0;
    for (int t = static_cast<int>(rewards.size()) - 1; t >= 0; t--) {
        cumulative = rewards[t] + gamma * cumulative;
        returns[t] = cumulative;
    }
    return returns;
}

std::vector<double> REINFORCE::normalizeReturns(const std::vector<double>& returns) const {
    if (returns.empty()) return {};
    double mean = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double r : returns) variance += (r - mean) * (r - mean);
    variance /= returns.size();
    double stdDev = std::sqrt(variance);
    if (stdDev < 1e-10) stdDev = 1.0;

    std::vector<double> normalizedReturns(returns.size());
    for (size_t i = 0; i < returns.size(); i++) {
        normalizedReturns[i] = (returns[i] - mean) / stdDev;
    }
    return normalizedReturns;
}

REINFORCE::REINFORCE(double discountFactor, double lr, unsigned int seed) 
    : environment(5, 5), gamma(discountFactor), learningRate(lr) {
    std::vector<int> architecture = {environment.getStateSize(), 16, environment.getActionSpace()};
    policy = std::make_unique<PolicyNetwork>(architecture, seed);
}

std::pair<std::vector<REINFORCE::Transition>, double> REINFORCE::runEpisode() {
    std::vector<Transition> trajectory;
    double totalReward = 0.0;
    auto state = environment.reset();
    bool done = false;

    while (!done) {
        int action = policy->sampleAction(state);
        auto [nextState, reward, isDone] = environment.step(action);
        trajectory.push_back({state, action, reward, isDone});
        state = nextState;
        done = isDone;
        totalReward += reward;
    }

    return {trajectory, totalReward};
}

void REINFORCE::updatePolicy(const std::vector<Transition>& trajectory) {
    if (trajectory.empty()) return;
    std::vector<double> rewards;
    for (const auto& trans : trajectory) rewards.push_back(trans.reward);
    auto returns = calculateReturns(rewards);
    auto normalizedReturns = normalizeReturns(returns);

    auto [currentWeights, currentBiases] = policy->getParameters();
    std::vector<std::vector<std::vector<double>>> weightGradients = currentWeights;
    std::vector<std::vector<double>> biasGradients = currentBiases;

    for (auto& layer : weightGradients)
        for (auto& neuron : layer)
            std::fill(neuron.begin(), neuron.end(), 0.0);
    for (auto& layer : biasGradients)
        std::fill(layer.begin(), layer.end(), 0.0);

    for (size_t t = 0; t < trajectory.size(); t++) {
        const auto& trans = trajectory[t];
        double returnValue = normalizedReturns[t];
        for (size_t layer = 0; layer < weightGradients.size(); layer++) {
            for (size_t neuron = 0; neuron < weightGradients[layer].size(); neuron++) {
                for (size_t input = 0; input < weightGradients[layer][neuron].size(); input++) {
                    weightGradients[layer][neuron][input] += 0.01 * trans.state[input % trans.state.size()] * returnValue;
                }
                biasGradients[layer][neuron] += 0.01 * returnValue;
            }
        }
    }

    policy->updateParameters(weightGradients, biasGradients, learningRate);
}

void REINFORCE::train(int numEpisodes, int renderFrequency) {
    std::vector<double> episodeRewards;
    for (int episode = 0; episode < numEpisodes; episode++) {
        auto [trajectory, totalReward] = runEpisode();
        updatePolicy(trajectory);
        episodeRewards.push_back(totalReward);

        int windowSize = std::min(100, episode + 1);
        double averageReward = std::accumulate(episodeRewards.end() - windowSize, episodeRewards.end(), 0.0) / windowSize;

        if (episode % 10 == 0 || episode == numEpisodes - 1) {
            std::cout << "Episode " << episode << ", Total Reward: " << totalReward 
                      << ", Average Reward (last " << windowSize << "): " << averageReward << std::endl;
        }

        if (renderFrequency > 0 && episode % renderFrequency == 0) {
            std::cout << "Rendering episode " << episode << std::endl;
            renderEpisode();
        }
    }
}

void REINFORCE::renderEpisode() {
    auto state = environment.reset();
    bool done = false;
    int step = 0;
    double totalReward = 0.0;

    std::cout << "Initial state:" << std::endl;
    environment.render();

    while (!done && step < 100) {
        int action = policy->sampleAction(state);
        auto [nextState, reward, isDone] = environment.step(action);
        totalReward += reward;

        std::cout << "Step " << step << ", Action: ";
        switch (action) {
            case 0: std::cout << "Up"; break;
            case 1: std::cout << "Right"; break;
            case 2: std::cout << "Down"; break;
            case 3: std::cout << "Left"; break;
        }
        std::cout << "\nReward: " << reward << ", Total: " << totalReward << std::endl;
        environment.render();

        state = nextState;
        done = isDone;
        step++;

        if (done) {
            std::cout << "Episode finished successfully in " << step << " steps!" << std::endl;
        } else {
            std::cout << "Episode reached step limit without completion." << std::endl;
        }
    }
}