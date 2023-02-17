import random
from common.population import Population
from common.non_dominated_sort import Non_Dominated_Sort
import numpy as np
from pymoo.factory import get_performance_indicator
from common.knapsack import Knapsack
from common.individual import Individual
from sklearn.ensemble import RandomForestRegressor
import copy


# Class to perform genetic operators
class Operators():

    def __init__(self,
                 problem):
        self.problem = problem


    def crossover_binary(self, individual1, individual2, population):
        population.last_id += 1
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()
        child1.id = population.last_id
        population.last_id += 1
        child2.id = population.last_id
        geneA = int(random.random() * len(individual1.features) - 2)
        geneB = int(random.random() * len(individual1.features) - 2)

        startGene = min(geneA, geneB)
        endGene = max(geneA, geneB)

        child1.trace = [0 for i in range(self.problem.num_of_variables)]
        child2.trace = [0 for i in range(self.problem.num_of_variables)]

        for i in range(0, startGene):
            child1.features[i] = individual1.features[i]
            child2.features[i] = individual2.features[i]

        for i in range(startGene, endGene):
            child1.features[i] = individual2.features[i]
            child2.features[i] = individual1.features[i]

        for i in range(endGene, len(individual1.features)):
            child1.features[i] = individual1.features[i]
            child2.features[i] = individual2.features[i]

        return child1, child2

    # Mutation for non binary problems
    def mutate_nonbinary(self, child):
        num_of_features = len(child.features)
        for gene in range(num_of_features):
            u = random.uniform(0, 1)
            prob = self.problem.mutation
            if u < prob:
                new_value = random.choice(self.problem.variables)
                while new_value == child.features[gene]:
                    new_value = random.choice(self.problem.variables)
                child.features[gene] = new_value

    # Mutation for binary problems
    def mutate_binary(self, child):
        num_of_features = len(child.features)
        for gene in range(num_of_features):
            u = random.uniform(0, 1)
            prob = self.problem.mutation
            if u < prob:
                if child.features[gene] == 1:
                    child.features[gene] = 0
                else:
                    child.features[gene] = 1


