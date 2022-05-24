import matplotlib.pyplot as plt
from genetic_algorithm import GA


def function(x, y, z):
    value = 3*x - 2*y + 10*z
    return value


if __name__ == '__main__':

    bounds = {
                'x': [-10, 10],
                'y': [-30, 30],
                'z': [0, 50] }

    initial_pop = [ [5, 10, 3],
                    [-3, 24, 32],
                    [9, 0, 0]]

    GeneticAlgorithm = GA(  generations_count=10,
                            population_count=20,
                            function=function,
                            params_bounds=bounds,
                            initial_population=initial_pop,
                            fitness_threshold=None,
                            maximize=False,
                            floating_point=True,
                            stochastic=False,
                            stochastic_iterations=3,
                            allow_gene_duplication=False,
                            crossover_percentage=0.3,
                            mutation_percentage=0.7 )

    GeneticAlgorithm.evolve( verbose=2 )

    result = GeneticAlgorithm.get_best()
    print('\nResult', result)

    best_fitness = GeneticAlgorithm.get_best_fitness()
    print('Best fitness', best_fitness)

    optymalization_time = GeneticAlgorithm.get_evolution_time()
    print('Optymalization time:', optymalization_time, 's\n')

    learning_curve = GeneticAlgorithm.plot_learning_curve()

    searched_list = GeneticAlgorithm.get_searched_list()
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in searched_list:
        ax.scatter(i[0], i[1], i[2], color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
