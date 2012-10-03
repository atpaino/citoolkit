import MultiLayerPerceptron as MLP
import random
import numpy as np

#Reads in a 4-d dataset from a file and builds a neural network
#classifier for it consisting of 1 hidden layer

#Read in dataset
data = []
f = open('two_class_4d_gaussian_data.txt', 'r+')
for line in f:
    try:
        #Parse this line of the file into 5 values separated by whitespace
        (w, x, y, z, cl) = line.rstrip().split()
    except:
        #No more valid data to read
        break

    #Convert features to floating point decimals
    w = float(w)
    x = float(x)
    y = float(y)
    z = float(z)

    #Convert the class to an integer
    cl = int(cl)

    #Add this datapoint (a list) to the dataset (also a list)
    data.append([w, x, y, z, cl])

#Shuffle dataset
random.shuffle(data)

#Divide dataset into training, validation, and testing subsets
training_data = data[:int(len(data)*.7)]
validation_data = data[int(len(data)*.7):int(len(data)*.85)]
testing_data = data[int(len(data)*.85):]

#Create MLP with default parameters
mlp = MLP.MultiLayerPerceptron(training_data, validation_data, testing_data, alpha=.00035, beta=.8, weight_init=lambda x,y : (2*np.random.rand(x,y)-1))

#Train mlp
num_iterations = mlp.do_training(max_iterations=15)

#Plot results
mlp.graph_results()
