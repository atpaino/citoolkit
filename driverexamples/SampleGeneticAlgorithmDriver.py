#Defines a fitness function for a trivial example problem
#and uses it to drive the EvolutionController class
import GenericGeneticAlgorithm as GGA

#Set up EvolutionController
ec = GGA.EvolutionController(GGA.GenericChromosome)

#Run algorithm
winner = ec.optimize()

print '\nMax fitness:', winner.fitness
for gene in winner.genes:
    print 'Gene:', gene.val
