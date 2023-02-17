from common.problem import Problem
from common.knapsack import Knapsack
from algorithms.moead import Moead
from algorithms.moead_nfts import Moead_Nfts
from algorithms.moead_rfts import Moead_Rfts
import random
from pathlib import Path
import threading

class IntMucop2():

    def __init__(self, array_combination):

        self.num_of_variables = array_combination[0]
        self.num_of_individuals = array_combination[1]
        self.algorithms = array_combination[2]
        self.tournament_size = array_combination[3]
        self.mutation_rate = array_combination[4]
        self.crossover_prob = array_combination[5]
        self.stopping_criteria = array_combination[7]
        self.spea2_archive_size = array_combination[6]
        self.generations = int((self.stopping_criteria / self.num_of_individuals))
        self.directions = ["max", "max"] if array_combination[8] == "max" else ["min", "min"]
        self.moead_neighborhood_size = array_combination[9]
        self.iteration = array_combination[10]
        self.criterion = array_combination[11]
        self.n_estimators = array_combination[12]
        self.max_depth = array_combination[13]
        self.min_samples_leaf = array_combination[14]
        self.max_features = array_combination[15]
        self.min_samples_split = array_combination[16]
        self.max_samples = array_combination[17]

        # begin execution
        self.do()

    def define_value_vector_knasapck(self, number):
        weights = []
        profits = []
        with open("instances/bin_int_mucop_" + str(self.num_of_variables) + "_" + str(number) + "_.txt", "rt") as f:
            capacity = int(f.readline())  # ignore first line
            for line in f:
                currentline = line.split(",")
                weights.append(int(currentline[0]))
                profits.append(int(currentline[1]))
        return capacity, weights, profits

    def get_profit(self, individual, capacity, weights, profits):
        weight, profit = 0, 0
        for (item, data) in zip(individual, profits):
            if item != 0:
                profit += data

        for (item, data) in zip(individual, weights):
            if item != 0:
                weight += data

        return profit

    def f1(self, individual):
        capacity, weights, profits = self.define_value_vector_knasapck(number=0)
        return self.get_profit(individual, capacity, weights, profits)

    def f2(self, individual):
        capacity, weights, profits = self.define_value_vector_knasapck(number=1)
        return self.get_profit(individual, capacity, weights, profits)



    def set_knap(self):
        weights, profits, capacities = [], [], []
        for i in range(3):
            capacity, weight, profit = self.define_value_vector_knasapck(number=i)
            weights.append(weight)
            profits.append(profit)
            capacities.append(capacity)
        knapClass = Knapsack(capacities, weights, profits)
        return knapClass

    def do(self):
        file_results = []

        problem = Problem(num_of_variables=self.num_of_variables,
                          num_of_individuals=self.num_of_individuals,
                          objectives=[self.f1, self.f2],
                          variables_range=[0, 5],
                          mutation=(1/self.num_of_variables),
                          discrete=True,
                          expand=False,
                          num_of_generations=self.generations,
                          directions=self.directions,
                          muco_nonbinary=True)

        nondominated = 0
        total_duplicated = []
        mean_error = []
        obj_func = []
        memoria = []
        tempoTotal = 0


        algorithm = self.algorithms
        # for algorithm in self.algorithms:
        #     algorithm = self.algorithms
        random.seed()
        if algorithm == "Moead":
            iteration = Moead(problem=problem,
                              neighborhood_size=self.moead_neighborhood_size)
            obj_func, tempoTotal, memoria, nondominated = iteration.run()
        elif algorithm == "Moead_Nfts":
            iteration = Moead_Nfts(problem=problem,
                              neighborhood_size=self.moead_neighborhood_size)
            obj_func, tempoTotal, memoria, total_duplicated, nondominated = iteration.run()
        elif algorithm == "Moead_Rfts" or algorithm == "Moead_Rfts_Worst":
            iteration = Moead_Rfts(problem=problem,
                               neighborhood_size=self.moead_neighborhood_size,
                               criterion=self.criterion,
                               max_depth=self.max_depth,
                               max_features=self.max_features,
                               min_samples_leaf=self.min_samples_leaf,
                               n_estimators=self.n_estimators,
                               min_samples_split=self.min_samples_split,
                               max_samples=self.max_samples)
            obj_func, tempoTotal, memoria, total_duplicated, mean_error, nondominated = iteration.run()


        list_x = []
        list_y = []


        for individual in nondominated:
            list_x.append(individual.objectives[0])
            list_y.append(individual.objectives[1])


        print("x_" + algorithm + "=" + str(list_x))
        print("y_" + algorithm + "=" + str(list_y))


        directory = "tests/results/" + algorithm + "/" + "int_mucop_2/"

        Path(directory).mkdir(parents=True, exist_ok=True)
        file_results = directory + algorithm + "_knap_2_" + str(self.num_of_variables) + "_iterations.txt"
        file_solutions = directory + algorithm + "_knap_2_" + str(self.num_of_variables) + "_solutions.txt"
        file_statistics = directory + algorithm + "_knap_2_" + str(self.num_of_variables) + "_statistics.txt"
        file_obj_gen = directory + algorithm + "_knap_2_" + str(self.num_of_variables) + "_igd_gen.txt"

        with open(file_results, "a+") as f:
            f.write("----Iteration " + str(self.iteration) + "----" + "\n")
            f.write("x_" + algorithm + "=" + str(list_x) + "\n")
            f.write("y_" + algorithm + "=" + str(list_y) + "\n")

        f.close()

        with open(file_obj_gen, "a+") as f:
            f.write("----Iteration " + str(self.iteration) + "----" + "\n")
            count = 0
            for element_gen in obj_func:
                string_gen = ""
                for element_obj in element_gen:
                    count_element = 0
                    for func in element_obj:
                        string_gen += (str(func))
                        if count_element == 0:
                            string_gen += (",")
                        count_element += 1
                    string_gen += (";")
                f.write(string_gen + "\n")
                count += 1

        f.close()

        with open(file_solutions, "a+") as f:
            f.write("----Iteration " + str(self.iteration) + "----" + "\n")
            for solution in nondominated:
                f.write(str(solution.features) + "\n")
        f.close()

        with open(file_statistics, "a+") as f:
            f.write("----Iteration " + str(self.iteration) + "----" + "\n")
            if algorithm == "Moead_Nfts" or algorithm == "Moead_Rfts":
                string = ','.join(map(str, total_duplicated))
                f.write("duplicated=" + string + "\n")

            if algorithm == "Moead_Rfts":
                string = ','.join(map(str, mean_error))
                f.write("mean_error=" + string + "\n")
            string = ','.join(map(str, memoria.tolist()))
            f.write("memory=" + string + "\n")
            string = ','.join(map(str, tempoTotal.tolist()))
            f.write("time=" + string + "\n")
        f.close()

        f = open(file_results)
        data = f.read()
        ocurrences = data.count("Iteration")
        if ocurrences == 10:
            with open(file_results, "a+") as f:
                f.write("----End of Run----" + "\n")
        f.close()








