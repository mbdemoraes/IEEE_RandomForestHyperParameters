import random
import copy

class NFTS:
    def __init__(self, problem):
        self.problem = problem
        self.tabu = set()


    def create_tabu_list(self, population):
        for individual in population:
            self.tabu.add(tuple(individual.features))


    # Local search for the BIN_MUCOP problem
    def local_search_unconstrained_mokp(self, child):
        items_to_remove = [i for i in range(len(child.features)) if child.features[i] == 1]
        items_to_add = [i for i in range(len(child.features)) if child.features[i] == 0]
        copy_child = copy.deepcopy(child)
        choice_remove = random.choice(items_to_remove)
        choice_add = random.choice(items_to_add)
        copy_child.features[choice_remove] = 0
        copy_child.features[choice_add] = 1
        items_to_remove.remove(choice_remove)
        items_to_add.remove(choice_add)
        child.features = copy_child.features
        return copy_child


    # Local search for the INT_MUCOP problem
    def local_search_muco_nonbinary(self, child):
        items_to_remove = [i for i in range(len(child.features)) if child.features[i] > min(self.problem.variables_range)]
        items_to_add = [i for i in range(len(child.features)) if child.features[i] < max(self.problem.variables_range)]
        copy_child = copy.deepcopy(child)
        choice_remove = random.choice(items_to_remove)
        choice_add = random.choice(items_to_add)
        copy_child.features[choice_remove] -= 1
        copy_child.features[choice_add] += 1
        items_to_remove.remove(choice_remove)
        items_to_add.remove(choice_add)
        child.features = copy_child.features
        return copy_child


    # Local search method for the BIN_MOKP problem for 2 and 3 objectives
    def local_search_mokp(self, child):
        is_infeasible = True
        if self.problem.num_of_objectives==2:
            items_to_remove = [i for i in range(len(child.features)) if child.features[i] == 1]
            items_to_add = [i for i in range(len(child.features)) if child.features[i] == 0]
            copy_child = copy.deepcopy(child)
            choice_remove = random.choice(items_to_remove)
            choice_add = random.choice(items_to_add)
            copy_child.features[choice_remove] = 0
            copy_child.features[choice_add] = 1
            items_to_remove.remove(choice_remove)
            items_to_add.remove(choice_add)
            is_infeasible = self.problem.knap.is_infeasible(copy_child, self.problem.num_of_objectives)
            #child.features = copy_child.features
        else:
            items_to_remove = [i for i in range(len(child.features)) if child.features[i] == 1]
            items_to_add = [i for i in range(len(child.features)) if child.features[i] == 0]
            copy_child = copy.deepcopy(child)
            choice_remove = random.choice(items_to_remove)
            choice_add = random.choice(items_to_add)
            copy_child.features[choice_remove] = 0
            copy_child.features[choice_add] = 1
            items_to_remove.remove(choice_remove)
            items_to_add.remove(choice_add)
            is_infeasible = self.problem.knap.is_infeasible(copy_child, self.problem.num_of_objectives)
            #child.features = copy_child.features
        return copy_child, is_infeasible