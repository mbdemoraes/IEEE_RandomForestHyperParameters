import copy

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from pandas import read_csv
from time import time
from numpy import vstack
from pathlib import Path
import os.path



def create_csv_results_tunning(combinations, means, test_name, lst_times):
    abspath = "tunning/results/"
    Path(abspath).mkdir(parents=True, exist_ok=True)
    filename = abspath + test_name + ".csv"

    if not os.path.exists(filename):
        with open(filename, "a+") as file:
            header = "n_estimators,max_depth,min_samples_split,min_samples_leaf,max_features,max_samples,"

            for i in range(len(means[0])):
                header += "mae_gen_" + str(i) + ","

            header += "final_mae,time"

            file.write(header + "\n")
    count = 0

    with open(filename, "a+") as file:
        for comb, mean, time in zip(combinations, means, lst_times):
            str_vector = ','.join(map(str, list(comb)))
            str_vector += ','
            str_obj = ','.join(map(str, mean))
            str_obj += ','
            final_row = str_vector + str_obj + str(np.mean(mean)) + "," + str(time)
            file.write(final_row + "\n")
            count += 1


abspath = "datasets/"
size_prob = 100
num_individuals = 120
test_name = "int_mucop_M3_" + str(size_prob)
name = test_name + ".csv"
filename = abspath + name
dataframe = read_csv(filename, header=None, skiprows=1)
data = dataframe.values
X, y = data[:, :-2], data[:, -2:]


#hyper-parameters values
n_estimators = [10,30,50,100]
max_depth = [10,30,50,None]
min_samples_split = [2,4,6,8]
min_samples_leaf = [1,2,4,8]
max_features = [0.4,0.6,0.8,1.0]
max_samples = [0.4,0.6,0.8,1.0]


mesh = np.array(np.meshgrid(n_estimators,
                            max_depth,
                            min_samples_split,
                            min_samples_leaf,
                            max_features,
                            max_samples))

#create all possible combinations
combinations = mesh.T.reshape(-1, 6)

final_means = []
lst_times = []
count = 0
# For each combination, do the training and prediction
for comb in combinations:
    print("Remaining: " + str(len(combinations) - count))
    means = [0] * 10
    time_means = []
    X_train = None
    y_train = None
    for k in range(0, 10):
        rf = RandomForestRegressor(n_estimators=int(comb[0]),
                                   max_depth=comb[1],
                                   min_samples_split=int(comb[2]),
                                   min_samples_leaf=int(comb[3]),
                                   max_features=comb[4],
                                   max_samples=comb[5],
                                   n_jobs=-1)
        #set as the training set the first num_individuals
        X_train = data[:num_individuals, :-2]
        y_train = data[:num_individuals, -2:]

        #train the RF
        rf.fit(np.array(X_train), np.array(y_train))

        count_gen = 0

        #Interactively test and train the RF, saving the mean MAE for each part
        for i in range(2,12,1):
            time_begin = time()
            X_test = data[num_individuals*(i-1):num_individuals*i, :-2]
            y_test = data[num_individuals*(i-1):num_individuals*i, -2:]

            predictions = rf.predict(np.array(copy.deepcopy(X_test)))
            X_train = vstack((X_train, copy.deepcopy(X_test)))
            y_train = vstack((y_train, copy.deepcopy(y_test)))
            rf.fit(np.array(X_train), np.array(y_train))
            errors = abs(predictions - np.array(y_test))  # Print out the mean absolute error (mae)
            means[count_gen] += (round(np.mean(errors), 2))
            time_end = time() - time_begin
            time_means.append(time_end)
            count_gen +=1
    means = [l/10 for l in means]
    final_mean = np.mean(means)
    #print("Final mean: " + str(final_mean))
    final_means.append(means)
    lst_times.append(np.mean(time_means))
    count+=1

print("Minimum final mean: " + str(min(final_means)))
test_name_results = test_name + "_tunning_results"
create_csv_results_tunning(combinations,final_means,test_name_results, lst_times)