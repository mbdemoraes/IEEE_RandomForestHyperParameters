from common.individual import Individual
import numpy as np
import copy
import random

class Knapsack:

    def __init__(self, capacities, weights, profits):
        self.capacities = capacities
        self.weights = weights
        self.profits = profits
        self.ratio = self.calc_ratios()

    # Calculate the ratio between profit and weight of each item
    def calc_ratios(self):
        ratios = []
        counter = 0
        for (weight, profit) in zip(self.weights, self.profits):
            ratios.append([])
            for (w, p) in zip(weight, profit):
                ratio = (p / w)
                ratios[counter].append(ratio)
            counter += 1
        return ratios

    # Return the cumulative weights of a solution
    def cumulative_weight(self, num_obj, features):
        cum_weights = [0 for i in range(num_obj)]
        obj_counter = 0
        for weight_list in self.weights:
            w_counter=0
            for weight in weight_list:
                if features[w_counter]==1:
                    cum_weights[obj_counter]+=weight
                w_counter+=1
            obj_counter+=1
        return cum_weights


    # Verify if it is an infeasible solution
    def is_infeasible(self, individual, num_of_objectives):
        cum_weights = self.cumulative_weight(num_of_objectives, individual.features)
        result = False
        counter = 0
        for capacity in self.capacities:
            if cum_weights[counter] > capacity:
                result = True
            counter+=1
        return result


    # Repair infeasible solutions
    def repair_infeasible(self, individual):
        cum_weights = self.cumulative_weight(len(individual.objectives), individual.features)
        ind_features = copy.deepcopy(individual.features)
        counter_ratios = 0
        new_cum_weights = cum_weights
        for i in range (len(individual.objectives)):
            if individual.objectives[i] == 0:
                rate = np.array(self.ratio[counter_ratios])
                size = int(len(individual.features)-(len(individual.features)*0.1))
                idx = np.argpartition(rate, size)
                smallest_idx = idx[:size]
                for index in smallest_idx:
                    result = False
                    for w,c in zip(new_cum_weights,self.capacities):
                        if w <= c:
                            result = True
                        else:
                            result = False
                            break
                    if result:
                        break
                    else:
                        if ind_features[index]==1:
                            ind_features[index]=0
                            new_cum_weights = self.cumulative_weight(len(individual.objectives), ind_features)
            counter_ratios+=1
        individual.features = copy.deepcopy(ind_features)
        return individual.features
