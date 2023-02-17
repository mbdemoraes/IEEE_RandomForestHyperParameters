import numpy as np


from itertools import  combinations

class MoeadUtils:

    def __init__(self,
                 problem):

        self.problem = problem

    def factorial(self, n):
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result

    def comb(self, n, k):
        return self.factorial(n) / (self.factorial(n - k) * self.factorial(k))

    #Simplex Lattice Design
    def UniformPoint(self, N, M):
        try:
            H1 = 1;
            while (self.comb(H1 + M, M - 1) <= N):
                H1 = H1 + 1

            temp1 = list(combinations(np.arange(H1 + M - 1), M - 1))
            temp1 = np.array(temp1)
            temp2 = np.arange(M - 1)
            temp2 = np.tile(temp2, (int(self.comb(H1 + M - 1, M - 1)), 1))
            W = temp1 - temp2
            W = (np.concatenate((W, np.zeros((np.size(W, 0), 1)) + H1), axis=1) - np.concatenate(
                (np.zeros((np.size(W, 0), 1)), W), axis=1)) / H1

            if H1 < M:
                H2 = 0
                while (self.comb(H1 + M - 1, M - 1) + self.comb(H2 + M, M - 1) <= N):
                    H2 = H2 + 1
                if H2 > 0:
                    temp1 = list(combinations(np.arange(H2 + M - 1), M - 1))
                    temp1 = np.array(temp1)
                    temp2 = np.arange(M - 1)
                    temp2 = np.tile(temp2, (int(self.comb(H2 + M - 1, M - 1)), 1))
                    W2 = temp1 - temp2
                    W2 = (np.concatenate((W2, np.zeros((np.size(W2, 0), 1)) + H2), axis=1) - np.concatenate(
                        (np.zeros((np.size(W2, 0), 1)), W2), axis=1)) / H2
                    W = np.concatenate((W, W2 / 2 + 1 / (2 * M)), axis=0)

            realN = np.size(W, 0)
            W[W == 0] = 10 ** (-6)
            if N!=realN:
                raise Exception("Population size unaivailable for the specified number of objectives.")
            return W, realN
        except Exception as error:
            print("Error while defining the weight vectors: " + repr(error))

    # Define the neighborhoods
    def FindNeighbour(self, W, N, M, T):
        B = []
        for i in range(N):
            temp = []
            for j in range(N):
                distance = 0
                for k in range(M):
                    distance += (W[i][k] - W[j][k]) ** 2
                distance = np.sqrt(distance)
                temp.append(distance)
            index = np.argsort(temp)
            B.append(index[:T])
        return B


    #set the initial reference point
    def find_best(self, population):
        z = [np.inf for i in range(len(population.population[0].objectives))]
        for individual in population:
            for i in range(len(z)):
                if individual.objectives[0] < z[i]:
                    z[i] = individual.objectives[0]
        return z

    #update reference point
    def update_reference_point(self, y, z):
        for j in range(len(z)):
            if self.problem.directions[j] == "max":
                if y.objectives[j] > z[j]:
                    z[j] = y.objectives[j]
            else:
                if y.objectives[j] < z[j]:
                    z[j] = y.objectives[j]

    def Tchebycheff(self, individual, weight, z):
        temp = []
        for i in range(len(individual.objectives)):
            temp.append(weight[i] * np.abs(individual.objectives[i] - z[i]))

        return np.max(temp)