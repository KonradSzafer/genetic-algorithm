# genetic-algorithm

## Documentation

### Instance initialization

```python
GeneticAlgorithm = GA(  generations_count=100,   # number of simulated generations
                        population_count=200,    # number of population in generation
                        function=function,       # function to optimize
                        params_bounds=bounds,    # variable bounds for each argument
                        maximise=False,          # decide to optimize values for minimising or maximising function output
                        floating_point=True,     # variables data type
                        stochastic=False,        # set True if function to optimize have stochastic nature
                        stochastic_iterations=3, # if function to optimize have stochastic nature, performs multiple calculations for every individual (>=3)
                        crossover_percentage=0.3,# percentage of population reproduced by crossover
                        mutation_percentage=0.7  # percentage of population reproduced by mutation
                        )
```

### Methods

```python
GeneticAlgorithm.evolve( verbose = True )
```
Runs a solution search.
If verbose = True, the progress will be displayed in console.

```python
GeneticAlgorithm.get_best()
```
Returns best solution.

```python
GeneticAlgorithm.get_best_fitness()
```
Returns best solution fitness.

```python
GeneticAlgorithm.get_evolution_time()
```
Returns searching time in seconds.

```python
GeneticAlgorithm.get_learning_curve()
```
Returns fitness array.

```python
GeneticAlgorithm.get_searched_list( include_fitness = False )
```
Returns list of searched solutions.
If include_fitness = True, the last element of solution is its fitness.