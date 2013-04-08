"""
This module contains code for various evolutionary computation algorithms, 
including a generic genetic algorithm along with its chromosome definition.
"""

import random
import numpy as np
import math
from multiprocessing import Process, Array

class GeneBase:
    """
    Represents a single gene in a chromosome. By default it is just
    a float between 0 and 1. This class is meant to be overwritten
    for more specific use cases.
    Also contains a method to control mutation of the gene.
    """

    def __init__(self):
        """
        Initializes this gene to a random float between 0 and 1.
        """
        self.val = np.random.rand()

    def mutate(self):
        """
        Randomly alters this gene.
        """
        self.val = math.floor(self.val + np.random.rand() - .5)

    def deep_copy(self, gene):
        """
        Performs a deep copy of the given gene onto this gene.
        """
        self.val = gene.val

class GenericChromosome:
    """
    Represents an individual in the populaltion for the algorithm.
    Contains a set of genes along with methods to mutate them.
    """

    def __init__(self, training_set, validation_set, testing_set, prob_mutation=.01, chromosome_size=10):
        """
        Initializes this chromosome by randomly creating a set of genes
        """
        self.prob_mutation = prob_mutation
        self.chromosome_size = chromosome_size
        self.genes = [ GeneBase() for i in range(0, chromosome_size) ]
        self.training_set = np.copy(training_set)
        self.validation_set = validation_set
        self.testing_set = testing_set

        #Get initial fitness score
        self.evaluate_fitness()

    def mutate(self):
        """
        Randomly alters a gene inside this chromosome.
        """
        for gene in self.genes:
            if np.random.rand() < self.prob_mutation:
                #Mutate the ith gene
                gene.mutate()

    def crossover(self, p2):
        """
        Produces two children from this chromosome and p2
        """
        #Initialize the children
        child1 = GenericChromosome(self.prob_mutation, self.chromosome_size)
        child2 = GenericChromosome(self.prob_mutation, self.chromosome_size)

        for i in range(0, len(self.genes)):
            if np.random.rand() > .5:
                child1.genes[i].deep_copy(self.genes[i])
                child2.genes[i].deep_copy(p2.genes[i])
            else:
                child2.genes[i].deep_copy(self.genes[i])
                child1.genes[i].deep_copy(p2.genes[i])

        return (child1, child2)


    def evaluate_fitness(self):
        """
        Defines the fitness function for this chromosome.
        Just a sample, trivial fitness function that tries to maximize
        the sum of the genes.
        """
        self.fitness = (sum([ gene.val for gene in self.genes ]) / self.chromosome_size)

class EvolutionController:
    """
    Primary class of algorithm which controls the flow of the
    genetic algorithm. Contains methods to initialize the population
    and begin the evolutionary process.
    Each Chromosome is of the form given by the GenericChromosome class above.
    The user need only modify/inherit this class to tailor the algorithm to their needs.
    training_set is assumed to be a list of training samples, with the label at the end of
    each sample.
    """

    def __init__(self, chromosome, training_set, validation_set, testing_set, prob_mutation=.01, prob_crossover=.5, population_size=100):
        """
        Sets various constants for the algorithm and initializes the
        population
        """
        self.chromosome = chromosome
        self.prob_mutation = prob_mutation
        self.prob_crossover = prob_crossover
        self.population_size = population_size

        #Initialize population
        self.population = [ chromosome(training_set, validation_set, testing_set, prob_mutation) for i in range(0, population_size) ]

    def optimize(self, generations=100, threshold=.99):
        """
        Performs iteration of the algorithm over the specified number
        of generations.
        """

        #Get initial fitness
        self.test_fitness()

        for i in range(0, generations):
            #Perform crossover by roulette wheel selection, replacing
            #the bottom half of the population with the offspring

            #Sort the population (descending order in terms of fitness)
            self.population.sort(key=lambda chromosome : chromosome.fitness, reverse=True)

            print 'Generation', (i+1), 'Max Fitness:', self.population[0].fitness,\
                    'Average Fitness: ', sum([c.fitness for c in self.population])/len(self.population)
            
            #Break if fitness is greater than a threshold value
            if self.population[0].fitness > threshold:
                break

            for j in range(self.population_size/2, self.population_size-1, 2):
                #Crossover based on probability of crossover
                if np.random.rand() > self.prob_crossover:
                    continue

                #Pick parent 1
                parent1 = j - (self.population_size/2)

                #Pick parent 2
                parent2 = self.pick_parent()
                while parent2 == parent1:
                    parent2 = self.pick_parent()

                #Crossover the parents
                (child1, child2) = self.population[parent1].crossover(self.population[parent2])
                self.population[j] = child1
                self.population[j+1] = child2

            self.test_fitness()

        #Sort the population (descending order in terms of fitness)
        self.population.sort(key=lambda chromosome : chromosome.fitness, reverse=True)

        #Return the optimal chromosome
        return (self.population[0], self.population[0].fitness)

    def test_fitness(self):
        #Create Array to hold returned fitness values
        arr = Array('d', range(len(self.population)))

        #Apply mutation to all population members and re-evaluate
        #their fitness in a multi-threaded fashion
        pool = []
        i = 0
        for chromo in self.population:
            #Mutate the chromosome first
            chromo.mutate()
            pool.append(Process(target=self.proc_eval_fitness, args=(chromo, arr, i)))
            i += 1
        for process in pool:
            process.start()
        i = 0
        for process in pool:
            process.join()   #Wait for all chromosomes to finish evaluating fitness
            self.population[i].fitness = arr[i]
            i += 1

    def proc_eval_fitness(self, chromosome, arr, i):
        """
        Just used to make the process of mutating and evaluating the fitness of a
        chromosome multithreaded.
        """
        arr[i] = chromosome.evaluate_fitness()
        if arr[i] > .95:
            print chromosome
            
            #Display results for this chromosome's top model
            chromosome.model.graph_results()
        
    def pick_parent(self):
        """
        Does roullete wheel selection using a random 'wheel spin'
        and returns the index of the chosen parent
        """

        #Pick 'rotation' through 'wheel'
        rotation = sum([ chromo.fitness for chromo in self.population ]) * np.random.rand()
        total_rotation = 0.0
        i = 0
        while total_rotation <= rotation:
            total_rotation += self.population[i].fitness
            i += 1

            #Check if we have wrapped around
            if i > self.population_size/2:
                i = i % self.population_size/2

        #Decrement i since we went past the chosen index
        return (i-1)
