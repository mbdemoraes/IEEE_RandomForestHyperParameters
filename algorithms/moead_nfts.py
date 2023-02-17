from common.non_dominated_sort import Non_Dominated_Sort
from common.moead_utils import MoeadUtils
import numpy as np
from common.population import Population
import os, psutil
import time
from offsprings.offspring_moead_nfts import OffspringMoeadNfts
from common.nfts import NFTS

class Moead_Nfts:

    # Implementation of the MOEA/D-NFTS algorithm based on the paper
    # M. B. de Moraes and G. P. Coelho, “A diversity preservation method
    # for expensive multi-objective combinatorial optimization problems using
    # Novel-First Tabu Search and MOEA/D,” Expert Systems with Applications,
    # vol. 202, p. 117251, sep 2022
    # DOI: https://doi.org/10.1016/j.eswa.2022.117251

    def __init__(self,
                 problem,
                 neighborhood_size):

        self.problem = problem
        self.nfts = NFTS(self.problem)
        self.offspring = OffspringMoeadNfts(self.problem, self.nfts)
        self.utils = MoeadUtils(problem)
        self.m = len(self.problem.objectives) #num_of_objectives
        self.population = None
        self.ndsort = Non_Dominated_Sort()
        self.external_population = Population()
        self.z = None
        self.visited_external = set()
        self.memory = np.zeros(self.problem.num_of_generations)
        self.time = np.zeros(self.problem.num_of_generations)
        self.obj_fun_gen = []
        self.neighborhood_size = neighborhood_size


    def run(self):
        # generate the set of weight vectors using the Simplex Lattice Design Method
        weights_vectors, self.problem.num_of_individuals= self.utils.UniformPoint(self.problem.num_of_individuals, self.m)

        # set the number of neighborhoods
        T = np.ceil(self.problem.num_of_individuals / self.neighborhood_size)
        T = int(T)

        # define the neighborhoods
        B = self.utils.FindNeighbour(weights_vectors, self.problem.num_of_individuals, self.m, T)

        # create an initial population
        self.population = self.problem.create_initial_population()

        # set the initial reference point
        self.z = self.utils.find_best(self.population)

        for i in range(self.problem.num_of_generations):
            inicio = time.time()
            print("Generation = " + str(i))
            #updates the tabu list
            self.nfts.create_tabu_list(self.population)
            self.ndsort.fast_nondominated_sort(self.population)
            #create the offspring while replacing the current individuals
            self.offspring.create_children_moead(self.population, B, T, self.z, weights_vectors, train_instances=None, train_labels=None, generation=i)

            # non-dominated sorting to store to the external population
            self.ndsort.fast_nondominated_sort(self.population)
            for individual in self.population.fronts[0]:
                if tuple(individual.features) not in self.visited_external:
                    self.visited_external.add(tuple(individual.features))
                    self.external_population.append(individual)
            lst = []
            self.ndsort.fast_nondominated_sort(self.external_population)
            for individual in self.external_population.fronts[0]:
                lst.append(individual.objectives)
            self.obj_fun_gen.append(lst)
            fim = time.time()
            self.time[i] = (fim-inicio)
            self.memory[i] = (psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        self.ndsort.fast_nondominated_sort(self.external_population)

        return self.obj_fun_gen, self.time, self.memory, self.offspring.total_duplicated, self.external_population.fronts[0]
