#include "../inc/node.hpp"
#include <cmath>

namespace yrgo {
namespace machine_learning {

/********************************************************************************
 * @brief Constructor for Node class.
 * @param num_weights Number of weights for the node.
 *******************************************************************************/
Node::Node(unsigned int num_weights) : weights_(num_weights) {
    InitRandomGenerator();
    bias_ = GetRandom();
    for (double& weight : weights_) {
        weight = GetRandom();
    }
    error_ = 0.0;
    output_ = 0.0;
}

/********************************************************************************
 * @brief Get the number of weights associated with this node.
 * @return The number of weights.
 *******************************************************************************/
unsigned int Node::GetNumWeights() const {
    return weights_.size();
}

/********************************************************************************
 * @brief Get the bias value of the node.
 * @return The bias value.
 *******************************************************************************/
double Node::GetBias() const {
    return bias_;
}

/********************************************************************************
 * @brief Get the current error of the node.
 * @return The error value.
 *******************************************************************************/
double Node::GetError() const {
    return error_;
}

/********************************************************************************
 * @brief Get the output value of the node.
 * @return The output value.
 *******************************************************************************/
double Node::Output() const {
    return output_;
}

/********************************************************************************
 * @brief Get the vector of weights associated with this node.
 * @return A constant reference to the weights vector.
 *******************************************************************************/
const std::vector<double>& Node::GetWeights() const {
    return weights_;
}

/********************************************************************************
 * @brief Perform a feedforward operation for the node based on input.
 * @param input The input vector.
 *******************************************************************************/
void Node::FeedForward(const std::vector<double>& input) {
    double sum = bias_;
    for (size_t i = 0; i < weights_.size(); ++i) {
        sum += input[i] * weights_[i];
    }
    output_ = std::max(0.0, sum); // Using ReLU activation function
}

/********************************************************************************
 * @brief Perform backpropagation on the node.
 * @param reference The reference value for backpropagation.
 *******************************************************************************/
void Node::Backpropagate(double reference) {
    error_ = reference - output_; // Calculate error based on reference value
}

/********************************************************************************
 * @brief Optimize the node's weights and bias using input and learning rate.
 * @param input The input vector.
 * @param learning_rate The learning rate for weight optimization.
 *******************************************************************************/
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
