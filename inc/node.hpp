/********************************************************************************
 * @brief Interface for implementation of neural nodes in machine learning 
 *        applications.
 ********************************************************************************/
#pragma once

#include <vector>
#include <cstdlib>
#include <ctime>

namespace yrgo {
namespace machine_learning {

/********************************************************************************
 * @brief Class for implementing neural nodes.
 ********************************************************************************/
class Node {
  public:

    /********************************************************************************
     * @brief Creates new neural node and initializes the bias and weights.
     * 
     * @param num_weighs
     *        The number of node weights.
     ********************************************************************************/
    Node(unsigned int num_weights);

    /********************************************************************************
     * @brief Provides the number of node weights.
     * 
     * @return
     *        The number of node weights as an unsigned integer.
     ********************************************************************************/
    unsigned int GetNumWeights() const;

    /********************************************************************************
     * @brief Provides the node bias.
     * 
     * @return
     *        The node bias as a floating-point number.
     ********************************************************************************/
    double GetBias() const;

    /********************************************************************************
     * @brief Provides the current node error.
     * 
     * @return
     *        The current node error as a floating-point number.
     ********************************************************************************/
    double GetError() const;

    /********************************************************************************
     * @brief Provides the node output.
     * 
     * @return
     *        The node output as a floating-point number.
     ********************************************************************************/
    double Output() const;

    /********************************************************************************
     * @brief Provides the node weights.
     * 
     * @return
     *        Reference to vector holding the node weights as floating-point numbers.
     ********************************************************************************/
    const std::vector<double>& GetWeights() const;

    /********************************************************************************
     * @brief Performs feedforward to update the node output.
     * 
     * @param input
     *        Reference to vector holding the new input values of the node.
     ********************************************************************************/
    void FeedForward(const std::vector<double>& input);

    /********************************************************************************
     * @brief Performs backpropagation to calculate the current node error.
     * 
     * @param reference
     *        The reference value, i.e. the desired value of the output.
     ********************************************************************************/
    void Backpropagate(double reference);

    /********************************************************************************
     * @brief Performs optimization by adjusting the bias and weights. This is done
     *        to improve the models accuracy.
     * 
     * @param input
     *        Reference to vector holding the current input values of the node.
     * @param learning_rate
     *        The learning rate to use for the optimization, i.e. how much of the 
     *        current error to adjust the node parameters with.
     ********************************************************************************/
    void Optimize(const std::vector<double>& input, double learning_rate);

  private:
    private:
    std::vector<double> weights_; /**< Vector holding the node weights */
    double bias_; /**< Node bias */
    double error_; /**< Current node error */
    double output_; /**< Node output */
    static constexpr double Relu(const double x) { return x > 0 ? x : 0; }
    static constexpr double ReluDelta(const double y) { return y > 0 ? 1 : 0; }

    static void InitRandomGenerator(void) { 
        static bool random_generator_initialized{false};
        if (!random_generator_initialized) {
            std::srand(std::time(nullptr));
            random_generator_initialized = true;
        }
    }

    static double GetRandom(const double min = 0, const double max = 1) { 
        return (std::rand() / static_cast<double>(RAND_MAX)) * (max - min) + min; 
    }
};

} /* namespace machine_learning */
} /* namespace yrgo */