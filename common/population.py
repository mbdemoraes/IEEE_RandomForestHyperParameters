class Population:

    def __init__(self):
        self.population = []
        self.last_id = 0
        self.var_count = []
        self.probs = []
        self.probs_cluster = []
        self.fronts = []
        self.distances = []

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):
        self.population.extend(new_individuals)

    def append(self, new_individual):
        self.population.append(new_individual)