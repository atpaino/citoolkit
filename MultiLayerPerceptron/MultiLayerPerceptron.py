import numpy as np
import matplotlib.pyplot as plt
import math
import random

class MultiLayerPerceptron:
    """
    A neural network with one hidden layer that includes
    functionality for training and testing a given dataset.
    The input data can be of any dimension, as long as all the 
    datapoints are floating point numbers and the classes are
    positive integers.
    Will produce n output nodes where n is the number of classes
    detected in the given data.
    """

    def __init__(self, training_data, validation_data, testing_data, num_hidden_nodes=None, alpha=.0004, beta=.5, weight_init=lambda x,y : np.random.randn(x, y), hidden_activation_fn=lambda x : np.tanh(x), output_activation_fn=lambda x : np.tanh(x), output_upper_threshold=.9, output_lower_threshold=-.9, hidden_activation_fn_deriv=lambda x : 1-np.tanh(x)**2, output_activation_fn_deriv=lambda x : 1-np.tanh(x)**2):
        """
        Constructor for the neural net. Sets various user-defined
        options, or to defaults if they are not given.
        """
        self.hidden_activation_fn = hidden_activation_fn
        self.output_activation_fn = output_activation_fn
        self.num_hidden_nodes = num_hidden_nodes
        self.training_data = training_data
        self.validation_data = validation_data
        self.testing_data = testing_data
        self.alpha = alpha
        self.beta = beta
        self.output_upper_threshold = output_upper_threshold
        self.output_lower_threshold = output_lower_threshold
        self.hidden_activation_fn_deriv = hidden_activation_fn_deriv
        self.output_activation_fn_deriv = output_activation_fn_deriv

        #Calculate dimensions of input data
        self.num_input_nodes = int(len(training_data[0])-1)
        self.num_output_nodes = 1
        for datapoint in training_data:
            if datapoint[-1] > self.num_output_nodes:
                self.num_output_nodes = datapoint[-1]
        self.num_output_nodes = int(self.num_output_nodes)
        
        #Default the number of hidden nodes to one more than the
        #number of input nodes
        if self.num_hidden_nodes is None:
            self.num_hidden_nodes = self.num_input_nodes+1
        self.num_hidden_nodes = int(self.num_hidden_nodes)

        #Create initial weights
        self.hidden_weights = weight_init(self.num_hidden_nodes, self.num_input_nodes+1)
        self.output_weights = weight_init(self.num_output_nodes, self.num_hidden_nodes+1)
        
        #Declare results lists
        self.training_results = []
        self.validation_results = []
        self.testing_results = []

    def do_training(self, max_iterations=100):
        """
        Trains the MLP until either the maximum number of iterations
        has been reached or the validation results stop improving
        """
        #Get a baseline test
        self.test_all_sets()

        for k in range(0, max_iterations):
            print k, self.validation_results[-1][0]
            self.train()
            self.test_all_sets()

            #Check to see if the validation set's results did not improve
            if len(self.validation_results) > 1 and self.validation_results[-1][0] >= self.validation_results[-2][0]:
                #Validation set has stopped improving, so end training
                pass

        #Return the number of iterations it took to converge (if it did at all)
        return k

    def train(self):
        """
        Performs on-line training of the MLP using backpropagation algorithm
        Returns False if we did not change any weights, True otherwise
        """
        #Shuffle training data
        hold = [ data for data in self.training_data ]
        random.shuffle(hold)
        self.training_data = hold

        #Declare arrays to hold (k-1)th weights
        prev_hidden_weights = np.array(self.hidden_weights)
        prev_out_weights = np.array(self.output_weights)

        #Iterate over all samples in training data
        for k in range(0, len(self.training_data)):
            dp = np.array(self.training_data[k])

            #Perform forward computation

            #Calculate hidden layer values
            v_hidden = [ self.calc_node_output(dp[0:-1], self.hidden_weights[i]) for i in range(0, self.num_hidden_nodes) ]
            y_hidden = [ self.hidden_activation_fn(v) for v in v_hidden ]

            #Calculate output layer values
            v_out = [ self.calc_node_output(y_hidden, self.output_weights[i]) for i in range(0, self.num_output_nodes) ]
            y_out = [ self.output_activation_fn(v) for v in v_out ]
            
            #Allocate error array
            err = np.zeros(self.num_output_nodes)

            #Calculate error in output layer
            for i in range(0, self.num_output_nodes):
                if (i+1) == dp[-1]:
                    #Desired output is upper bound
                    err[i] = self.output_upper_threshold - y_out[i]
                else:
                    #Desired output is lower bound
                    err[i] = self.output_lower_threshold - y_out[i]
                    
            #Calculate gradients
            output_gradients = [ err[i]*self.output_activation_fn_deriv(v_out[i]) for i in range(0, self.num_output_nodes) ]
            hidden_gradients = [ self.hidden_activation_fn_deriv(v_hidden[i])*sum([ output_gradients[j]*self.output_weights[j][i] for j in range(0, self.num_output_nodes) ]) for i in range(0, self.num_hidden_nodes) ]

            #Save current weights so we can update previous weights arrays later
            curr_hidden_weights = np.array(self.hidden_weights)
            curr_out_weights = np.array(self.output_weights)

            #Adjust hidden weights
            for j in range(0, self.num_hidden_nodes):
                for i in range(0, self.num_input_nodes):
                    self.hidden_weights[j][i] = curr_hidden_weights[j][i] + self.beta*(curr_hidden_weights[j][i] - prev_hidden_weights[j][i]) + self.alpha*hidden_gradients[j]*dp[i]

                #Update weight for bias term
                self.hidden_weights[j][-1] = curr_hidden_weights[j][-1] + self.beta*(curr_hidden_weights[j][-1] - prev_hidden_weights[j][-1]) + self.alpha*hidden_gradients[j]

            #Adjust output weights
            for j in range(0, self.num_output_nodes):
                for i in range(0, self.num_hidden_nodes):
                    self.output_weights[j][i] = curr_out_weights[j][i] + self.beta*(curr_out_weights[j][i] - prev_out_weights[j][i]) + self.alpha*output_gradients[j]*y_hidden[i]

                #Update weight for bias term
                self.output_weights[j][-1] = curr_out_weights[j][-1] + self.beta*(curr_out_weights[j][-1] - prev_out_weights[j][-1]) + self.alpha*output_gradients[j]

            #Update previous weight arrays
            prev_hidden_weights = curr_hidden_weights
            prev_out_weights = curr_out_weights
    
    def predict_class(self, dp):
        """
        Accepts an input (feature) vector and returns the class predicted
        by this MLP
        """
        #Compute hidden layer values
        hidden_vals = [ self.hidden_activation_fn(self.calc_node_output(dp, self.hidden_weights[i])) for i in range(0, self.num_hidden_nodes) ]

        #Compute output layer values
        output_vals = np.zeros(self.num_output_nodes)
        max_val_ind = 0 #Keep track of the index of the maximum output value
        for i in range(0, self.num_output_nodes):
            #Calculate output value of this node
            output_vals[i] = self.output_activation_fn(self.calc_node_output(hidden_vals, self.output_weights[i]))

            #Update maximum output value
            if output_vals[i] > output_vals[max_val_ind]:
                max_val_ind = i

        #Return the class predicted
        return max_val_ind

    def test_all_sets(self):
        """
        Convenience method to test all three datasets on the MLP
        """
        self.training_results.append(self.test_data(self.training_data))
        self.validation_results.append(self.test_data(self.validation_data))
        self.testing_results.append(self.test_data(self.testing_data))

    def test_data(self, data):
        """
        Tests the given data against the MLP and returns the results
        """
        #Set various counters
        num_total = 0
        num_correct = 0
        total_err = 0.0

        #Initialize confusion matrix
        confusion_matrix = np.zeros([self.num_output_nodes, self.num_output_nodes])

        #Iterate through each point in this dataset
        for dp in data:
            #Compute hidden layer values
            hidden_vals = [ self.hidden_activation_fn(self.calc_node_output(dp[0:-1], self.hidden_weights[i])) for i in range(0, self.num_hidden_nodes) ]

            #Create flag to check if this datapoint was correctly matched
            correct = True

            #Compute output layer values
            output_vals = np.zeros(self.num_output_nodes)
            err = np.zeros(self.num_output_nodes)
            max_val_ind = 0 #Keep track of the index of the maximum output value
            for i in range(0, self.num_output_nodes):
                #Calculate output value of this node
                output_vals[i] = self.output_activation_fn(self.calc_node_output(hidden_vals, self.output_weights[i]))

                #Calculate error in output layer
                if (i+1) == dp[-1]:
                    #Desired output is upper bound
                    err[i] = (self.output_upper_threshold - output_vals[i])**2
                else:
                    #Desired output is lower bound
                    err[i] = (self.output_lower_threshold - output_vals[i])**2
                    
                #Update maximum output value
                if output_vals[i] > output_vals[max_val_ind]:
                    max_val_ind = i

            #Check if this value was classified correctly
            correct = dp[-1]-1 == max_val_ind

            #Update aggregate error
            total_err += sum(err)
            
            #Update confusion matrix
            confusion_matrix[max_val_ind][dp[-1]-1] += 1

            #Increment counters
            num_correct += 1 if correct else 0
            num_total += 1
            
        #Return a tuple of the counters, which represents the results
        return (total_err/float(2*num_total), num_total, num_correct, confusion_matrix)

    def calc_node_output(self, input_vals, weights):
        """
        Calculates the output of a given node by summing the dot product of its weights
        with its input values plus the bias term (last value in weights)
        Returns this as a float
        """
        #Set value to zero initially
        output = 0.0

        #Add dot product of weights with input values
        for i in range(0, len(input_vals)):
            output += input_vals[i] * weights[i]

        #Add bias term
        output += weights[-1]

        return output

    def graph_results(self):
        """
        Plots results including graph of errors vs. epoch and 
        confusion matrices for each of the datasets
        """
        #Print out confusion matrices
        print '\nTraining classification percentage: ', 100.0 * self.training_results[-1][2] / self.training_results[-1][1]
        print 'Training confusion matrix (size=', self.training_results[-1][1], ')'
        print self.training_results[-1][3] / self.training_results[-1][1]

        print '\nValidation classification percentage: ', 100.0 * self.validation_results[-1][2] / self.validation_results[-1][1]
        print 'Validation confusion matrix (size=', self.validation_results[-1][1], ')'
        print self.validation_results[-1][3] / self.validation_results[-1][1]

        print '\nTesting classification percentage: ', 100.0 * self.testing_results[-1][2] / self.testing_results[-1][1]
        print 'Testing confusion matrix (size=', self.testing_results[-1][1], ')'
        print self.testing_results[-1][3] / self.testing_results[-1][1]

        #Plot training, validation, and testing results
        plt.plot([ result[0] for result in self.training_results ])
        plt.plot([ result[0] for result in self.validation_results ])
        plt.plot([ result[0] for result in self.testing_results ])

        #Add legend
        plt.legend(('Training', 'Validation', 'Testing'), 'upper center', shadow=True, fancybox=True)

        #Add axes labels
        plt.xlabel('Epoch')
        plt.ylabel('Mean-squared Error')

        #Add title to figure
        plt.title("Learning rate: {}, Momentum: {}, Hidden units: {}".format(self.alpha, self.beta, self.num_hidden_nodes))

        #Display plot
        plt.show()
