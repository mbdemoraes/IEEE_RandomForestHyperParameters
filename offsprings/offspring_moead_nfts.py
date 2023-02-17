import numpy as np
from common.moead_utils import MoeadUtils
from common.operators import Operators
from common.population import Population
from common.individual import Individual
from numpy import vstack

class OffspringMoeadNfts:

    def __init__(self, problem, nfts, rf=None):
        self.problem = problem
        self.genetic_operators = Operators(self.problem)
        self.moead_utils = MoeadUtils(self.problem)
        self.nfts = nfts
        self.rf = rf
        self.total_duplicated = np.zeros(self.problem.num_of_generations)
        self.total_entered = 0


    def create_children_moead(self, population, B, T, z, weight_vector, train_instances, train_labels,generation=None):
        children_features = []
        children_labels = []
        child_pop = Population()
        for i in range(self.problem.num_of_individuals):
            #select two parents from the neighborhood
            k = np.random.randint(0, T)
            l = np.random.randint(0, T)
            while k == l:
                l = np.random.randint(0, T)
            pop_parent = [population.population[B[i][k]], population.population[B[i][l]]]

            child, child2 = self.genetic_operators.crossover_binary(pop_parent[0], pop_parent[1], population)

            # Only perform this if the algorithm is MOEA/D-RFTS
            if self.rf:
                # if parameters have the same decision vector
                # do local search
                if pop_parent[0].features == pop_parent[1].features:
                    self.rf.local_search_rf(child, i, population, B, T)
                else:
                    # otherwise perform mutation
                    if self.problem.muco_nonbinary:
                        self.genetic_operators.mutate_nonbinary(child)
                    else:
                        self.genetic_operators.mutate_binary(child)
            else:
                #if it is not the MOEA/D-RFTS, go direct to the mutation
                if self.problem.muco_nonbinary:
                    self.genetic_operators.mutate_nonbinary(child)
                else:
                    self.genetic_operators.mutate_binary(child)

            #Only perform this if the algorithm is MOEA/D-NFTS
            if self.nfts.tabu and not self.rf:
                if tuple(child.features) in self.nfts.tabu:
                    is_infeasible = True
                    self.total_duplicated[generation]+=1
                    while tuple(child.features) in self.nfts.tabu and is_infeasible:
                        #if problem is BIN_MOKP
                        if self.problem.knap:
                            child, is_infeasible = self.nfts.local_search_mokp(child)
                        # if problem is BIN_MUCOP
                        elif self.problem.knap_unconstrained:
                            child = self.nfts.local_search_unconstrained_mokp(child)
                            is_infeasible = False
                        # if problem is INT_MUCOP
                        elif self.problem.muco_nonbinary:
                            child = self.nfts.local_search_muco_nonbinary(child)
                            is_infeasible = False


            self.problem.calculate_objectives(child)
            self.problem.repair(child)
            children_features.append(child.features)
            children_labels.append(child.objectives)
            ind = Individual(directions=self.problem.directions)
            ind.objectives = child.objectives
            ind.features = child.features
            child_pop.append(ind)

            self.moead_utils.update_reference_point(child, z)

            entered = False
            for j in range(T):
                if self.moead_utils.Tchebycheff(child, weight_vector[B[i][j]], z) < self.moead_utils.Tchebycheff(population.population[B[i][j]],
                                                                                     weight_vector[B[i][j]], z):
                    population.population[B[i][j]] = child
                    if self.rf and child.came_from==True and not entered:
                        self.total_entered +=1
                        self.total_duplicated[generation] +=1
                        entered = True

        if self.rf:
            train_instances = vstack((train_instances, children_features))
            train_labels = vstack((train_labels, children_labels))
            self.rf.predict_and_train(train_instances, train_labels, generation)
            print('Total entered:' + str(self.total_entered))
