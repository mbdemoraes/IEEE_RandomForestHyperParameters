from common.non_dominated_sort import Non_Dominated_Sort
from common.moead_utils import MoeadUtils
import numpy as np
from offsprings.offspring_moead import OffspringMoead
from common.population import Population
import os, psutil
import time

class Moead:

    #Implementation of MOEA/D algorithm based on the paper
    #Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition,"
    #in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731,
    #Dec. 2007, doi: 10.1109/TEVC.2007.892759.

    def __init__(self,
                 problem,
                 neighborhood_size):

        #method's hyper-parameters
        self.problem = problem
        self.offspring = OffspringMoead(problem=problem)
        self.utils = MoeadUtils(problem)
        self.num_of_generations = problem.num_of_generations
        self.num_of_individuals = problem.num_of_individuals
        self.m = len(self.problem.objectives) #num_of_objectives
        self.population = None
        self.external_population = Population()
        self.ndsort = Non_Dominated_Sort()
        self.z = None
        self.visited_external = set()
        self.memory = np.zeros(problem.num_of_generations)
        self.time = np.zeros(problem.num_of_generations)
        self.obj_fun_gen = []
        self.neighborhood_size = neighborhood_size


    def run(self):
        #generate the set of weight vectors using the Simplex Lattice Design Method
        weights_vectors, self.num_of_individuals= self.utils.UniformPoint(self.num_of_individuals, self.m)

        #set the number of neighborhoods
        T = np.ceil(self.num_of_individuals / self.neighborhood_size)
        T = int(T)

        #define the neighborhoods
        B = self.utils.FindNeighbour(weights_vectors, self.num_of_individuals, self.m, T)

        #create an initial population
        self.population = self.problem.create_initial_population()

        #set the initial reference point
        self.z = self.utils.find_best(self.population)

        for i in range(self.num_of_generations):
            inicio = time.time()
            print("Generation = " + str(i))
            #create the offspring
            self.offspring.create_children_moead(self.population, B, T, self.z, weights_vectors)

            #non-dominated sorting to store to the external population
            self.ndsort.fast_nondominated_sort(self.population)
            for individual in self.population.fronts[0]:
                if tuple(individual.features) not in self.visited_external:
                    self.visited_external.add(tuple(individual.features))
                    self.external_population.append(individual)
            lst = []
            self.ndsort.fast_nondominated_sort(self.external_population)
            for individual in self.external_population.fronts[0]:
                lst.append(individual.objectives)

            self.external_population.population = self.external_population.fronts[0]
            self.obj_fun_gen.append(lst)
            fim = time.time()
            self.time[i] = (fim - inicio)
            self.memory[i] = (psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        self.ndsort.fast_nondominated_sort(self.external_population)

        return  self.obj_fun_gen, self.time, self.memory, self.external_population.fronts[0]
