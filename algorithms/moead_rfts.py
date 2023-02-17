from common.non_dominated_sort import Non_Dominated_Sort
from common.moead_utils import MoeadUtils
import numpy as np
from common.population import Population
import os, psutil
import time
from offsprings.offspring_moead_nfts import OffspringMoeadNfts
from common.nfts import NFTS
from common.random_forests import RANDOM_FORESTS

class Moead_Rfts:

    # Implementation of the MOEA/D-RFTS algorithm based on the paper
    # M. B. De Moraes and G. P. Coelho, “A random forest-assisted
    # decomposition-based evolutionary algorithm for multi-objective combinatorial optimization problems,”
    # in 2022 IEEE Congress on Evolutionary Computation (CEC), pp. 1–8, 2022

    def __init__(self,
                 problem,
                 neighborhood_size,
                 criterion,
                 max_depth,
                 max_features,
                 min_samples_leaf,
                 n_estimators,
                 min_samples_split,
                 max_samples):

        self.problem = problem
        self.nfts = NFTS(self.problem)
        self.rf = RANDOM_FORESTS(self.problem, self.nfts,
                                 criterion=criterion,
                                 max_depth=max_depth,
                                 max_features=max_features,
                                 min_samples_leaf=min_samples_leaf,
                                 n_estimators=n_estimators,
                                 min_samples_split=min_samples_split,
                                 max_samples=max_samples)
        self.offspring = OffspringMoeadNfts(self.problem, self.nfts, self.rf)
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
        self.train_instances = []
        self.train_labels = []



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
        lst = []
        lst_x = []
        lst_y = []
        self.ndsort.fast_nondominated_sort(self.population)
        for individual in self.population.fronts[0]:
            lst.append(individual.objectives)
            lst_x.append(individual.objectives[0])
            lst_y.append(individual.objectives[1])
        self.obj_fun_gen.append(lst)

        # add the initial population as training data
        train_features = []
        train_labels = []
        for individual in self.population:
            train_features.append(individual.features)
            train_labels.append(individual.objectives)
        self.train_instances = np.array(train_features)
        self.train_labels = np.array(train_labels)

        # train the RF with the initial population data
        self.rf.fit_population(np.array(train_features), np.array(train_labels))

        # set the initial reference point
        self.z = self.utils.find_best(self.population)

        for i in range(self.problem.num_of_generations):
            inicio = time.time()
            print("Generation = " + str(i))
            # Create/Update a tabu list
            self.nfts.create_tabu_list(self.population)
            self.ndsort.fast_nondominated_sort(self.population)

            # Create the offspring solutions
            self.offspring.create_children_moead(self.population, B, T, self.z, weights_vectors, self.train_instances, self.train_labels, generation=i)
            self.ndsort.fast_nondominated_sort(self.population)
            for individual in self.population.fronts[0]:
                # add the solution to the external population only if it has not been added yet
                if tuple(individual.features) not in self.visited_external:
                    self.visited_external.add(tuple(individual.features))
                    self.external_population.append(individual)

            # identify the non-dominated solutions of the current generation
            lst = []
            lst_x = []
            lst_y = []
            self.ndsort.fast_nondominated_sort(self.external_population)
            for individual in self.external_population.fronts[0]:
                lst.append(individual.objectives)
                lst_x.append(individual.objectives[0])
                lst_y.append(individual.objectives[1])

            # add the non-dominated solutions of the current generation
            self.obj_fun_gen.append(lst)

            # get the time and memory used
            fim = time.time()
            self.time[i] = (fim-inicio)
            self.memory[i] = (psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

        self.ndsort.fast_nondominated_sort(self.external_population)

        return self.obj_fun_gen, self.time, self.memory, self.offspring.total_duplicated, self.rf.mean_error, self.external_population.fronts[0]
