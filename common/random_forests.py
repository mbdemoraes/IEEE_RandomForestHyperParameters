import numpy as np
from common.population import Population
import copy
import random
from common.nfts import NFTS
from sklearn.ensemble import RandomForestRegressor

class RANDOM_FORESTS:

    def __init__(self, problem, nfts,
                 n_estimators,
                 criterion,
                 max_depth,
                 max_features,
                 min_samples_leaf,
                 min_samples_split,
                 max_samples):
        self.problem = problem
        self.nfts = nfts
        self.mean_error = np.zeros(self.problem.num_of_generations)
        #self.random_forest_regressor = RandomForestRegressor(n_estimators = 500, random_state = 42)
        self.random_forest_regressor = RandomForestRegressor(n_estimators=n_estimators,
                                                             max_depth=max_depth,
                                                             max_features=max_features,
                                                             min_samples_leaf=min_samples_leaf,
                                                             min_samples_split=min_samples_split,
                                                             criterion=criterion,
                                                             max_samples=max_samples,
                                                             bootstrap=True,
                                                             n_jobs=8)

    def get_center_vector(self, i, population, B, T):
        individual = []
        center_vector = [0] * self.problem.num_of_objectives
        for k in range(T):
            individual = population.population[B[i][k]]
            for l in range(len(individual.objectives)):
                center_vector[l] += individual.objectives[l]
                #center_vector[l]+= np.mean(self.mape)
        center_vector = [(i/T) for i in center_vector]
        return center_vector


    def local_search_rf(self, child, ind_now, population, B, T):
        is_infeasible = True
        children = Population()
        set_forbidden = set()
        set_forbidden.add(tuple(child.features))
        childs_features = []
        center_vector = self.get_center_vector(ind_now, population, B, T)
        distances = []
        for k in range(self.problem.num_of_individuals):
            copy_child = copy.deepcopy(child)
            while (tuple(copy_child.features) in self.nfts.tabu and is_infeasible == True) or tuple(copy_child.features) in set_forbidden :
                if self.problem.knap:
                    copy_child, is_infeasible = self.nfts.local_search_mokp(copy_child)
                elif self.problem.knap_unconstrained:
                    copy_child = self.nfts.local_search_unconstrained_mokp(copy_child)
                    is_infeasible = False
                elif self.problem.muco_nonbinary:
                    copy_child = self.nfts.local_search_muco_nonbinary(copy_child)
                    is_infeasible = False


            children.append(copy_child)
            childs_features.append(copy_child.features)
            set_forbidden.add(tuple(copy_child.features))

        predictions = self.random_forest_regressor.predict(np.array(childs_features))  # Calculate the absolute errors
        for pred, calc_child in zip(predictions, children):
            calc_child.objectives = [int(i) for i in pred]

        for calc_child in children:
            distance = np.linalg.norm(np.array(calc_child.objectives) - np.array(center_vector))
            distances.append(distance)

        # Find the closest estimation to the center vector
        index_min = min(range(len(distances)), key=distances.__getitem__)
        child.features = children.population[index_min].features
        child.came_from = True

    def fit_population(self, train_features, train_labels):
        self.random_forest_regressor.fit(np.array(train_features), np.array(train_labels))

    def predict_and_train(self, children_features, children_labels, generation=None):
        predictions = self.random_forest_regressor.predict(np.array(children_features))
        self.random_forest_regressor.fit(np.array(children_features), np.array(children_labels))
        errors = abs(predictions - np.array(children_labels))  # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
        self.mean_error[generation] = round(np.mean(errors), 2)
