import numpy as np
from common.moead_utils import MoeadUtils
from common.operators import Operators

class OffspringMoead:

    def __init__(self, problem):
        self.problem = problem
        self.genetic_operators = Operators(self.problem)
        self.moead_utils = MoeadUtils(self.problem)

    def create_children_moead(self, population, B, T, z, weight_vector):
        for i in range(self.problem.num_of_individuals):
            # select two parents from the neighborhood
            k = np.random.randint(0, T)
            l = np.random.randint(0, T)

            #avoid replacement
            while k == l:
                l = np.random.randint(0, T)
            pop_parent = [population.population[B[i][k]], population.population[B[i][l]]]

            #crossover
            child, child2 = self.genetic_operators.crossover_binary(pop_parent[0], pop_parent[1], population)

            #mutation
            self.genetic_operators.mutate_binary(child)

            #calculate the objective functions
            self.problem.calculate_objectives(child)

            # repair if necessary
            self.problem.repair(child)

            y = child

            # update reference point
            self.moead_utils.update_reference_point(y, z)

            #apply the Tchebycheff decomposition
            for j in range(T):
                if self.moead_utils.Tchebycheff(y, weight_vector[B[i][j]], z) < self.moead_utils.Tchebycheff(population.population[B[i][j]],
                                                                                     weight_vector[B[i][j]], z):
                    population.population[B[i][j]] = y


