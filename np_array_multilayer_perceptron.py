import random
import math
import numpy as np

# Multilayer Perceptron with 1 hidden layer
class NpArrayMultilayerPerceptron:
    # Default learning rate, number of layers and tuple with the amount of units/neurons per layer
    def __init__(self, learning_rate, units_per_layer_tuple, expected_output):
        
        # Learning rate for the neural network
        self.learning_rate = learning_rate
        # Input layer
        self.input_layer_values = np.zeros(shape=(units_per_layer_tuple[0]))
        # Weights and bias between input layer and hidden layer, randomized between -0.5 and 0.5
        self.input_hidden_layer_weights = np.random.rand(units_per_layer_tuple[1], units_per_layer_tuple[0]) - np.array([0.5])
        self.hidden_layer_biases = np.random.rand(units_per_layer_tuple[1]) - np.array([0.5])
        # Sum before evaluation on hidden layer units/neurons
        self.hidden_layer_sums = np.zeros(shape=(units_per_layer_tuple[1]))
        # Hidden layer values
        self.hidden_layer_values = np.zeros(shape=(units_per_layer_tuple[1]))
        # Weights and bias between hidden layer and output layer, randomized between -0.5 and 0.5
        self.hidden_output_layer_weights = np.random.rand(units_per_layer_tuple[2], units_per_layer_tuple[1]) - np.array([0.5])
        self.output_layer_biases = np.random.rand(units_per_layer_tuple[2]) - np.array([0.5])
        # Sum before evaluation on output layer units/neurons
        self.output_layer_sums = np.zeros(shape=(units_per_layer_tuple[2]))
        # Output layer values
        self.output_layer_values = np.zeros(shape=(units_per_layer_tuple[2])) 
        # Expected results in output units
        self.expected_output = expected_output

    # Introduce feature vector to input layer
    def introduce_data_to_input_layer(self, feature_vector):
        self.input_layer_values = feature_vector

    # Transfer data from the input layer to hidden layer
    def transfer_data_to_hidden_layer(self):    
        # Iterate hidden layer units
        for i in range(0, len(self.input_hidden_layer_weights)):
            # A unit/neuron from the previous layer is connected to all units/neurons of the next layer (each connection is a weight)
            # Also, a unit/neuron from the next layer is connected to all units/neurons of the previous layer (each connection is a weight)
            neuron_sum = np.dot(self.input_layer_values, self.input_hidden_layer_weights[i])
            neuron_sum += self.hidden_layer_biases[i]
            # Stores the neuron sum before the evaluation
            self.hidden_layer_sums[i] = neuron_sum
            # Neuron function will evaluate the its sum with a defined function
            evaluate_output = self.feedforward_evaluation(neuron_sum)
            self.hidden_layer_values[i] = evaluate_output

    # Transfer data from the hidden layer to output layer
    def transfer_data_to_output_layer(self):    
        # Iterate output layer units
        for i in range(0, len(self.hidden_output_layer_weights)):
            # A unit/neuron from the previous layer is connected to all units/neurons of the next layer (each connection is a weight)
            # Also, a unit/neuron from the next layer is connected to all units/neurons of the previous layer (each connection is a weight)
            neuron_sum = np.dot(self.hidden_layer_values, self.hidden_output_layer_weights[i])
            neuron_sum += self.output_layer_biases[i]
            # Stores the neuron sum before the evaluation
            self.output_layer_sums[i] = neuron_sum
            # Neuron function will evaluate the its sum with a defined function
            evaluate_output = self.feedforward_evaluation(neuron_sum)
            self.output_layer_values[i] = evaluate_output

    # The feed-forward currently used function is the logistic function
    def feedforward_evaluation(self, value):
        return 1/(1 + math.exp(-value))

    # Derivate logistic to compute errors
    def derivate_logistic(self, value):
        return math.exp(value)/((math.exp(value)+1)**2)

    # Compute error by calculating offset from expected value (a letter, in this case) and adjust weights via backpropagation
    # Also returns total error from output units
    def compute_error_and_adjust_weights(self, expected_value):
        k_error_list, output_error_list = self.compute_error_from_output_layer(expected_value)
        # The output layer id (index) would be 2, so our multilayer perceptron, with one hidden layer, has its hidden layer id as 1
        #previous_layer_id = 1
        j_error_list = self.compute_error_from_hidden_layer(k_error_list)
        # Adjust weights between output layer and the hidden layer
        #layer_id = 2
        self.adjust_weights_in_output_layer(k_error_list)
        # Adjust weights between hidden layer and the input layer
        #layer_id = 1
        self.adjust_weights_in_hidden_layer(j_error_list)
        # Returns total error from output units
        return output_error_list

     # Adjust weights between input layer and hidden layer
    def adjust_weights_in_hidden_layer(self, error_list):
        np_error_list = np.array(error_list)
        np_learning_rate = np.array(self.learning_rate) 
         # Adjust weights between input layer and hidden layer
        for i in range(len(self.input_hidden_layer_weights)):
            # Increment weights with gradients
            self.input_hidden_layer_weights[i] += np_learning_rate * self.input_layer_values * np_error_list[i]
        # Adjust biases between layer_index and (layer_index-1)
        # Increment bias with gradients
        
        self.hidden_layer_biases += np_learning_rate * np_error_list

    # Adjust weights between hidden layer and output layer
    def adjust_weights_in_output_layer(self, error_list):
        np_error_list = np.array(error_list)
        np_learning_rate = np.array(self.learning_rate)
        # Adjust weights between hidden layer and output layer
        for i in range(len(self.hidden_output_layer_weights)):
            # Increment weights with gradients
            self.hidden_output_layer_weights[i] +=  np_learning_rate * self.hidden_layer_values * np_error_list[i]
        # Adjust biases between layer_index and (layer_index-1)
        # Increment bias with gradients
        self.output_layer_biases += np_learning_rate * np_error_list

    # Compute error from hidden layers
    # k_error_list would be the list of errors for each neuron that came up from the lower/next layer
    def compute_error_from_hidden_layer(self, k_error_list):
        j_error_list = []
        # Iterate neurons in previous_layer_id
        for i in range(len(self.input_hidden_layer_weights)):
            neuron_error_sum = 0
            # Iterate neurons in previous_layer_id + 1
            for j in range(len(self.hidden_output_layer_weights)):
                neuron_error_sum += self.input_hidden_layer_weights[i][j] * k_error_list[j]
            # Neuron error mount = neuron error sum * derivate neuron sum
            neuron_error_amount = neuron_error_sum * self.derivate_logistic(self.hidden_layer_sums[i])
            j_error_list.append(neuron_error_amount)
        return j_error_list

    # Compute error from output layer and return error per output unit
    def compute_error_from_output_layer(self, expected_value):
        k_error_list = []
        output_error_list = []
        # This value will correspond to each output unit expected output
        expected_output_from_neuron = 0
        # Amount of error per output unit
        error_amount = 0
        # Iterate output layer units
        for i in range(len(self.output_layer_values)):
            # Compare if the the neuron was supposed or not to identify the letter from expected_value
            if(expected_value == self.expected_output[i]):
                expected_output_from_neuron = 1
            else:
                expected_output_from_neuron = 0
            # Error amount = (expected value from output unit - actual value from output unit) * derivate neuron sum
            output_neuron_error = (expected_output_from_neuron - self.output_layer_values[i])
            error_amount = output_neuron_error * self.derivate_logistic(self.output_layer_sums[i])
            k_error_list.append(error_amount)
            output_error_list.append(output_neuron_error)
        # Return list with all the error from output units
        return k_error_list, output_error_list

    # Return a dictionary with the letters as keys (ex.: "Z") and its correspondent output neuron value (from 0 to 1)
    def predict(self, feature_vector):
        # Introduce the feature vector to the input layer
        self.introduce_data_to_input_layer(feature_vector)
        # Transfer from input to hidden layer, and eventually from hidden layer to the output layer
        self.transfer_data_to_hidden_layer()
        self.transfer_data_to_output_layer()
        predictions = {}
        output_layer_index = len(self.output_layer_values)-1
        for i in range(len(self.output_layer_values)):
            predictions[self.expected_output[i]] = self.output_layer_values[i]
        return predictions