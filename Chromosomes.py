import numpy as np

class AbstractChromosome:
    """
    Defines the methods that each chromosome must implement
    """

    def __init__(self, training_set, validation_set, testing_set, prob_mutation=.01, x_rate=.5):
        """
        The constructor method for any chromosome must have this signature (with
        possibly different default values for prob_mutation and x_rate)
        """
        pass

    def mutate(self):
        """
        This methods randomly modifies the genes in this chromosome.
        """
        pass

    def crossover(self, p2):
        """
        This method defines how crossover between two chromosomes should occur.
        Returns a tuple containing the two children produced, which are of the
        same type as the parents.
        """
        pass
    
    def evaluate_fitness(self):
        """
        This method tests the fitness of this chromosome and returns the fitness
        which must be in the interval [0,1]
        """
        pass
