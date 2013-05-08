import numpy as np
import matplotlib.pyplot as plt
import math
import random

class MultiLayerPerceptron:
    """
    A neural network (trained via backpropagation) with a specifiable number of
    hidden layers, each with a specifiable number of nodes.
    that includes functionality for training and testing a given dataset.
    The input data can be of any dimension, as long as all the 
    datapoints are floating point numbers and the classes are
    positive integers.
    Will produce n output nodes where n is the number of classes
    detected in the given data.
    NOTE: the labels for each datapoint must start at 1 (i.e. a dataset with
    two classes would have either 1 or 2 for its label). If a 0 is present as a label,
    the initialization method will just increment the class of each datapoint.
    """

    def __init__(self, training_feats, training_labels,
            validation_feats, validation_labels, testing_data, testing_labels,
            alpha=.0004, beta=.5, hidden_layer_structure=None,
            weight_init=lambda x,y : np.random.randn(x, y), 
            activation_fn=lambda x : np.tanh(x), 
            output_upper_threshold=.9, output_lower_threshold=-.9, 
            activation_fn_deriv=lambda x : 1-np.tanh(x)**2, 
        """
        Constructor for the neural net. Sets various user-defined
        options, or to defaults if they are not given.
        Note: assumes the labels are integers starting from 0.
        Parameters:
            alpha := learning rate
            beta := momentum rate
            hidden_layer_structure := np array specifying number of hidden
                nodes in each hidden layer, where len(hidden_layer_structure)
                is the number of hidden layers.
            weight_init := function that returns matrix of random values,
                where the size of the matrix is the input argument set
        """
        self.alpha = alpha
        self.beta = beta
        self.hidden_layer_structure = hidden_layer_structure
        self.activation_fn = activation_fn
        self.activation_fn_deriv = activation_fn_deriv
        self.training_feats = np.array(training_feats)
        self.validation_feats = np.array(validation_feats)
        self.testing_feats = np.array(testing_feats)
        self.training_labels = np.array(training_labels)
        self.validation_labels = np.array(validation_labels)
        self.testing_labels = np.array(testing_labels)
        self.output_upper_threshold = output_upper_threshold
        self.output_lower_threshold = output_lower_threshold

        #Calculate number of classes of input data
        self.num_input_nodes = len(training_feats[0])
        self.num_output_nodes = int(training_labels.max()
        
        #Default the number of hidden nodes to one more than the
        #number of input nodes, and just one hidden layer
        if hidden_layer_structure is None:
            self.hidden_layer_structure = np.array(self.num_input_nodes+1)

        #Create array containing number of nodes for each layer
        self.layer_sizes = np.insert(self.hidden_layer_structure, 0, 
                self.num_input_nodes)
        self.layer_sizes = np.append(self.layer_sizes, self.num_output_nodes)
        self.num_layers = len(self.layer_sizes)

        #Initialize weights s.t. weights[i,j,k] is the weight from the kth node
        #in layer i to the jth node in layer i+1. 1 is added to layer_sizes[i]
        #here to account for the weight from the bias (1 for each node/layer).
        self.weights = np.array([weight_init(int(self.layer_sizes[i+1]),
                                 int(self.layer_sizes[i]+1))
                                 for i in xrange(self.num_layers-1)])
        
        #Declare results lists
        self.training_results = []
        self.validation_results = []
        self.testing_results = []

    def do_training(self, max_iterations=100, min_iterations=0, verbose=False):
        """
        Trains the MLP until either the maximum number of iterations
        has been reached or the validation results stop improving
        """
        #Get a baseline test
        self.test_all_sets()

        for k in range(0, max_iterations):
            if verbose:
                print (k+1), self.validation_results[-1][0]
            self.train()
            self.test_all_sets()

            #Check to see if the validation set's results did not improve
            if (len(self.validation_results) > 1 and 
                    self.validation_results[-1][0] >= self.validation_results[-2][0] 
                    and k >= (min_iterations-1)):
                #Validation set has stopped improving, so end training
                break

        #Return the maximum classification percentage reached
        return self.testing_results[-1][1]

    #TODO not refactored yet
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
            v_hidden = [ self.calc_node_output(dp[0:-1], self.hidden_weights[i])
                    for i in range(0, self.num_hidden_nodes) ]
            y_hidden = [ self.hidden_activation_fn(v) for v in v_hidden ]

            #Calculate output layer values
            v_out = [ self.calc_node_output(y_hidden, self.output_weights[i]) 
                    for i in range(0, self.num_output_nodes) ]
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
            output_gradients = [err[i]*self.output_activation_fn_deriv(v_out[i]) 
                    for i in range(0, self.num_output_nodes)]
            hidden_gradients = [self.hidden_activation_fn_deriv(v_hidden[i])*
                    sum([output_gradients[j]*self.output_weights[j][i] 
                    for j in range(0, self.num_output_nodes)]) 
                    for i in range(0, self.num_hidden_nodes)]

            #Save current weights so we can update previous weights arrays later
            curr_hidden_weights = np.array(self.hidden_weights)
            curr_out_weights = np.array(self.output_weights)

            #Adjust hidden weights
            for j in range(0, self.num_hidden_nodes):
                for i in range(0, self.num_input_nodes):
                    self.hidden_weights[j][i] = (curr_hidden_weights[j][i] +
                            self.beta*(curr_hidden_weights[j][i] - 
                                prev_hidden_weights[j][i]) + 
                            self.alpha*hidden_gradients[j]*dp[i])

                #Update weight for bias term
                self.hidden_weights[j][-1] = (curr_hidden_weights[j][-1] + 
                        self.beta*(curr_hidden_weights[j][-1] - 
                            prev_hidden_weights[j][-1]) + 
                        self.alpha*hidden_gradients[j])

            #Adjust output weights
            for j in range(0, self.num_output_nodes):
                for i in range(0, self.num_hidden_nodes):
                    self.output_weights[j][i] = (curr_out_weights[j][i] + 
                            self.beta*(curr_out_weights[j][i] - 
                                prev_out_weights[j][i]) + 
                            self.alpha*output_gradients[j]*y_hidden[i])

                #Update weight for bias term
                self.output_weights[j][-1] = (curr_out_weights[j][-1] + 
                        self.beta*(curr_out_weights[j][-1] - 
                            prev_out_weights[j][-1]) + 
                        self.alpha*output_gradients[j])

            #Update previous weight arrays
            prev_hidden_weights = curr_hidden_weights
            prev_out_weights = curr_out_weights
    
    def predict_proba(self, dp):
        """
        Accepts an input (feature) vector and returns an array containing the
        output from each output node.
        """
        #Initialize next_layer with feature vector
        next_layer = dp

        for i in xrange(self.num_layers-1):
            #Calculate inputs to layer i+1. 1 is appended to the feature
            #vector to represent the bias. Applies activation fn after.
            next_layer = self.activation_fn(np.matrix(np.append(next_layer,
                1.0))*np.matrix(self.weights[i]).transpose())

        return next_layer

    def test_all_sets(self):
        """
        Convenience method to test all three datasets on the MLP
        """
        self.training_results.append(self.test_data(self.training_feats, 
            self.training_labels))
        self.validation_results.append(self.test_data(self.validation_feats,
            self.validation_labels))
        self.testing_results.append(self.test_data(self.testing_feats,
            self.testing_labels))

    def test_data(self, data, labels):
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
        for i, dp in enumerate(data):
            #Compute output layer values
            outputs = self.predict_proba(dp)
            err = np.zeros(self.num_output_nodes)
            #Keep track of the index of the maximum output value
            max_val_ind = 0

            for i in xrange(0, self.num_output_nodes):
                #Calculate error in output layer
                if (i+1) == dp[-1]:
                    #Desired output is upper bound
                    err[i] = (self.output_upper_threshold - outputs[i])**2
                else:
                    #Desired output is lower bound
                    err[i] = (self.output_lower_threshold - outputs[i])**2
                    
                #Update maximum output value
                if outputs[i] > outputs[max_val_ind]:
                    max_val_ind = i

            #Check if this value was classified correctly
            correct = labels[i] == max_val_ind

            #Update aggregate error
            total_err += err.sum()
            
            #Update confusion matrix
            confusion_matrix[max_val_ind][labels[i]] += 1

            #Increment counters
            num_correct += 1 if correct else 0
            num_total += 1
            
        #Return a tuple of the counters, which represents the results
        return (total_err/float(2*num_total), float(num_correct)/num_total, 
                num_total, num_correct, confusion_matrix)

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

    def graph_results(self, show_graph=False):
        """
        Plots results including graph of errors vs. epoch and 
        confusion matrices for each of the datasets
        """
        #Print out confusion matrices
        print '\nTraining classification percentage: ', 100.0 * self.training_results[-1][1]
        print 'Training confusion matrix (size=', self.training_results[-1][2], ')'
        print self.training_results[-1][4] / self.training_results[-1][2]

        print '\nValidation classification percentage: ', 100.0 * self.validation_results[-1][1]
        print 'Validation confusion matrix (size=', self.validation_results[-1][2], ')'
        print self.validation_results[-1][4] / self.validation_results[-1][2]

        print '\nTesting classification percentage: ', 100.0 * self.testing_results[-1][1]
        print 'Testing confusion matrix (size=', self.testing_results[-1][2], ')'
        print self.testing_results[-1][4] / self.testing_results[-1][2]

        if show_graph:
            #Plot training, validation, and testing results
            plt.plot([ result[0] for result in self.training_results ])
            plt.plot([ result[0] for result in self.validation_results ])
            plt.plot([ result[0] for result in self.testing_results ])

            #Add legend
            plt.legend(('Training', 'Validation', 'Testing'), 'upper center', 
                    shadow=True, fancybox=True)

            #Add axes labels
            plt.xlabel('Epoch')
            plt.ylabel('Mean-squared Error')

            #Add title to figure
            plt.title("Learning rate: {}, Momentum: {}, Hidden units: {}".format(
                self.alpha, self.beta, self.num_hidden_nodes))

            #Display plot
            plt.show()

class NeuralFusionClassifier:
    """
    A "meta" classifier that trains a set of neural nets using different
    parameters (number of nodes, features used) and then trains one more
    neural net that fuses the output from the neural net set. The feature set
    given to the fusing neural net includes the confidence values from each
    neural net along with the complete original feature vector (in order to
    allow for context-awareness).
    """
    pass

def split_dataset(data):
    """
    Splits the dataset up into training, validation, and testing subsets.
    """
    training_data = data[:int(len(data)*.7)]
    validation_data = data[int(len(data)*.7):int(len(data)*.85)]
    testing_data = data[int(len(data)*.85):]

    return (training_data, validation_data, testing_data)
