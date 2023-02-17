from common.individual import Individual
import random
import numpy as np
from common.population import Population
import copy

# Class that configures the problem
class Problem:

    def __init__(self,
                 objectives,
                 num_of_variables,
                 variables_range,
                 num_of_individuals,
                 directions,
                 num_of_generations,
                 mutation,
                 expand=True,
                 discrete=False,
                 knap=None,
                 knap_unconstrained=None,
                 muco_nonbinary=None):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.num_of_individuals = num_of_individuals
        self.objectives = objectives
        self.expand = expand
        self.discrete = discrete
        self.variables_range = variables_range
        self.knap = knap
        self.directions = directions
        self.num_of_generations = num_of_generations
        self.variables = self.set_variables()
        self.mutation = mutation
        self.knap_unconstrained = knap_unconstrained
        self.muco_nonbinary = muco_nonbinary

    def set_variables(self):
        variables = [i for i in range(min(self.variables_range), max(self.variables_range) + 1)]
        return variables


    def create_initial_population(self):
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.generate_individual()
            individual.id = _
            individual.trace = [_ for i in range(self.num_of_variables)]
            self.calculate_objectives(individual)
            self.repair(individual)
            population.append(individual)
            population.last_id = _
        return population


    def generate_individual(self):
        individual = Individual(self.directions)
        individual.features = [random.randint(min(self.variables_range), max(self.variables_range)) for x in range(self.num_of_variables)]
        return individual

    def repair(self, individual):
        try:
            if self.knap:
                while 0 in individual.objectives:
                    self.knap.repair_infeasible(individual)
                    self.calculate_objectives(individual)
        except Exception as ex:
            print(ex)


    def calculate_objectives(self, individual):
        if self.expand:
            individual.objectives = [f(*individual.features) for f in self.objectives]
        else:
            individual.objectives = [f(individual.features) for f in self.objectives]

