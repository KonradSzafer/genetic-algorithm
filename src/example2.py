import time
import math
import numpy as np
import matplotlib.pyplot as plt
from genetic_algorithm import GA


X = [0, 10, 20, 25, 30, 33, 35, 38, 39, 40]
Y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def fitting_equation(w, x):
    y = w[0]*x**3 + w[1]*x**2 + w[2]*x
    y += w[3]*(math.e**(w[4]*x))
    y += w[5]*x**w[6]
    return y


def function(w0, w1, w2, w3, w4, w5, w6):

    fittnes = 0
    for i in range(len(X)):
        x = X[i]
        pred_y = fitting_equation([w0, w1, w2, w3, w4, w5, w6], x)
        true_y = Y[i]
        fittnes += abs(true_y - pred_y)

    return fittnes


if __name__ == '__main__':

    bounds = {
                'w0': [-1, 1],
                'w1': [-1, 1],
                'w2': [-1, 1],
                'w3': [-1, 1],
                'w4': [-1, 1],
                'w5': [-1, 1],
                'w6': [0, 1]}

    GeneticAlgorithm = GA(  generations_count=1000,
                            population_count=1000,
                            function=function,
                            params_bounds=bounds,
                            fitness_threshold=None,
                            maximize=False,
                            floating_point=True,
                            stochastic=False,
                            stochastic_iterations=3,
                            allow_gene_duplication=True,
                            crossover_percentage=0.2,
                            mutation_percentage=0.6 )

    GeneticAlgorithm.evolve( verbose=0 )

    result = GeneticAlgorithm.get_best()
    print('\nResult', result)

    best_fitness = GeneticAlgorithm.get_best_fitness()
    print('Best fitness', best_fitness)

    optymalization_time = GeneticAlgorithm.get_evolution_time()
    print('Optymalization time:', optymalization_time, 's\n')

    learning_curve = GeneticAlgorithm.plot_learning_curve()

    new_y = []
    for i in range(len(X)):
        x = X[i]
        y = fitting_equation(result, x)
        new_y.append(y)

    plt.plot(X, Y)
    plt.plot(X, new_y)
    plt.show()
