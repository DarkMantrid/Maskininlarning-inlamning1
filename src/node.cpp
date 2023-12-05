#include "../inc/node.hpp"
#include <cmath>

namespace yrgo {
namespace machine_learning {

Node::Node(unsigned int num_weights) : weights_(num_weights) {
    InitRandomGenerator();
    bias_ = GetRandom();
    for (double& weight : weights_) {
        weight = GetRandom();
    }
    error_ = 0.0;
    output_ = 0.0;
}


unsigned int Node::GetNumWeights() const {
    return weights_.size();
}

double Node::GetBias() const {
    return bias_;
}

double Node::GetError() const {
    return error_;
}

double Node::Output() const {
    return output_;
}

const std::vector<double>& Node::GetWeights() const {
    return weights_;
}

void Node::FeedForward(const std::vector<double>& input) {
    double sum = bias_;
    for (size_t i = 0; i < weights_.size(); ++i) {
        sum += input[i] * weights_[i];
    }
    output_ = 1.0 / (1.0 + std::exp(-sum)); // Using sigmoid activation function
}

void Node::Backpropagate(double reference) {
    error_ = reference - output_; // Calculate error based on reference value
}

void Node::Optimize(const std::vector<double>& input, double learning_rate) {
    // Update weights and bias using backpropagation-derived formulas
    for (size_t i = 0; i < weights_.size(); ++i) {
        // Update each weight using the gradient descent algorithm
        weights_[i] += learning_rate * error_ * input[i];
    }
    // Update bias using the gradient descent algorithm
    bias_ += learning_rate * error_;
}


} /* namespace machine_learning */
} /* namespace yrgo */
