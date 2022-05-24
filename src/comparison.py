import time
import numpy as np
import matplotlib.pyplot as plt

from genetic_algorithm import GA


def rosenbrock_function(x, y):
    a, b = 1, 100
    z = (a - x)**2 + b*(y - x**2)**2
    return z


def rosenbrock_function_scipy(xy):
    a, b = 1, 100
    z = (a - xy[0])**2 + b*(xy[1] - xy[0]**2)**2
    return z


if __name__ == '__main__':

    x_min, x_max = -10, 10
    x_num = x_max + 1 - x_min
    x = np.linspace(x_min, x_max, num=x_num)

    y_min, y_max = -100, 100
    y_num = y_max + 1 - y_min
    y = np.linspace(y_min, y_max, num=y_num)

    X, Y = np.meshgrid(x, y)
    Z = rosenbrock_function(X, Y)

    # Genetic Algorithm
    params_ranges = {
                    'x': (-10, 10),
                    'y': (-100, 100)}

    GeneticAlgorithm = GA(  generations_count=15,
                            population_count=15,
                            function=rosenbrock_function,
                            parameters_ranges=params_ranges,
                            maximise=False,
                            floating_point=True,
                            stochastic=False,
                            stochastic_iterations=3,
                            crossover_percentage=0.3,
                            mutation_percentage=0.7 )

    GeneticAlgorithm.evolve( verbose=True )

    result = GeneticAlgorithm.get_best()
    print('\nResult', result)

    best_fitness = GeneticAlgorithm.get_best_fitness()
    print('Best fitness', best_fitness)

    optymalization_time = GeneticAlgorithm.get_evolution_time()
    print('Optymalization time:', optymalization_time, 's\n')

    learning_curve = GeneticAlgorithm.get_learning_curve()
    plt.plot(learning_curve)
    plt.show()

    searched_list = GeneticAlgorithm.get_searched_list(include_fitness=True)

    # Plot
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z,
                    rstride=1,
                    cstride=1,
                    cmap='winter',
                    edgecolor='none')

    for i in searched_list:
        ax.scatter(i[0], i[1], i[2], color='black')

    ax.scatter(result[0], result[1], best_fitness, s=100, color='red')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
