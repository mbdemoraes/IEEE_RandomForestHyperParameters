# The number in the class name indicates the number of objective (2 or 3)

from tests.test_bin_mokp_2 import BinMokp2
from tests.test_bin_mokp_3 import BinMokp3
from tests.test_int_mucop_2 import IntMucop2
from tests.test_int_mucop_3 import IntMucop3
from tests.test_bin_mucop_2 import BinMucop2
from tests.test_bin_mucop_3 import BinMucop3
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool

#optimization hyper-parameters
list_num_of_variables = [100]
list_tournament_size = [2]
list_crossover_probs = [1.0]
list_num_of_individuals = [100]
list_mutation_rates = [(1/list_num_of_individuals[0])]
list_archive_size = [100,150,250]
list_neighborhood_size = [int(list_num_of_individuals[0]*0.1)]

# The iterations are independent from each other, therefore it is possible
# to parallelize them
list_iterations = [0,1,2,3,4,5,6,7,8,9]
algorithms = ["Moead"]
criterion = ["squared_error"]
n_estimators = [10]
max_depth = [10]
min_samples_split = [2]
min_samples_leaf = [8]
max_features = [0.4]
max_samples = [0.4]

stopping_criteria = [list_num_of_individuals[0]*10] #number of objective function evaluations
directions = ["max"] #direction of the problem, max for maximization or min for minimization


mesh = np.array(np.meshgrid(list_num_of_variables,
                            list_num_of_individuals, algorithms,
                            [None], list_mutation_rates,
                            [None],[None],
                            stopping_criteria,
                            directions, list_neighborhood_size,
                            list_iterations,
                            criterion,
                            n_estimators,
                            max_depth,
                            min_samples_leaf,
                            max_features,
                            min_samples_split,
                            max_samples))

#creates all possible combinations of the hyper-parameters
#if the algorithm does not use a given hyper-parameter, it will be igored during the optimization
combinations = mesh.T.reshape(-1, 18)


# generate threads to perform parallel computation
# the number is a user-input parameter equivalent to the number of available cores
# in your CPU
num_threads = 8 if len(combinations) >= 8 else len(combinations)
#num_threads = 1 # if you want to run in serial
pool = ThreadPool(num_threads)


# parallelize the test functions
# function map receives a function with a specific array combination
# change the first parameter (in this example BinMokp3) accordingly to the desired test function
results = pool.map(BinMucop2, combinations)

# closes the pool
pool.close()
pool.join()
