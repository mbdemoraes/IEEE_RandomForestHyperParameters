## Code to investigate the effects of the Random Forest Hyper-Parameters in the surrogate modeling of multi-objective combinatorial optimization problems
![Graphical abstract](/images/GraphicalAbstract.png)


Source code and supplementary material for the manuscript
"Effects of the Random Forests Hyper-Parameters in Surrogate Models for Multi-Objective Combinatorial Optimization: A Case Study using MOEA/D-RFTS"
 published in the IEEE Latin America Transactions.

### Datasets

The datasets for the benchmark problems are in the folder ```datasets```.

### Optimization
To perform optimization, run the script ```optimization.py```.
The results of the optimization procedures are saved on the folder
```tests/results```.

### Grid search for the Random Forest Hyper-Parameters

To investigate the performance of a Random Forest with a specific set of hyper-parameter values on the datasets, run the script ```rf_tunning.py```.
The results are saved on the folder ```results/tunning```.

### Contact
Any comments, suggestions or doubts, feel free to contact Matheus Bernardelli de moraes at m121214@dac.unicamp.br