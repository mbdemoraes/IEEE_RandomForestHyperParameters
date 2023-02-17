class Individual(object):

    def __init__(self, directions):
        self.id = 0
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.num_dominated = None
        self.dominated_solutions = None
        self.features = None
        self.objectives = None
        self.strength = None
        self.raw_fitness = None
        self.spea_fitness = None
        self.repeated = False
        self.trace = []
        self.crossover_gen = 0
        self.came_from = None
        self.hv = None
        self.utility = 1
        self.convergence = None
        self.diversity = None
        self.directions = directions


    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False


    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        final_condition = False
        # if len(self.directions)==2:
        #     if self.directions[0]=="max":
        #         for first, second in zip(self.objectives, other_individual.objectives):
        #             and_condition = and_condition and first >= second
        #             or_condition = or_condition or first > second
        #     else:
        #         for first, second in zip(self.objectives, other_individual.objectives):
        #             and_condition = and_condition and first <= second
        #             or_condition = or_condition or first < second
        #
        #     if self.directions[0] == "max" and self.directions[1] == "min":
        #         if (self.objectives[0] >= other_individual.objectives[0] and self.objectives[1] < other_individual.objectives[1]) or (self.objectives[0] > other_individual.objectives[0] and self.objectives[1] <= other_individual.objectives[1]):
        #             and_condition = True
        #             or_condition = True
        #         else:
        #             and_condition = False
        #             or_condition = False
        # else:
        conditions = []
        for k in range(len(self.directions)):
            if self.directions[k]== "max":
                if self.objectives[k] >= other_individual.objectives[k]:
                    conditions.append(True)
                else:
                    conditions.append(False)
            else:
                if self.objectives[k] <= other_individual.objectives[k]:
                    conditions.append(True)
                else:
                    conditions.append(False)

        if False not in conditions:
            for k in range(len(self.directions)):
                if self.directions[k] == "max":
                    if self.objectives[k] > other_individual.objectives[k]:
                        final_condition = True
                        break
                else:
                    if self.objectives[k] < other_individual.objectives[k]:
                        final_condition = True
                        break
        else:
            final_condition = False

        return final_condition











