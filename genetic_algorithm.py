import random
import math
import copy
import numpy as np

default_problems = {
    5: [(733, 251), (706, 87), (546, 97), (562, 49), (576, 253)],
    10: [
        (470, 169),
        (602, 202),
        (754, 239),
        (476, 233),
        (468, 301),
        (522, 29),
        (597, 171),
        (487, 325),
        (746, 232),
        (558, 136),
    ],
    12: [
        (728, 67),
        (560, 160),
        (602, 312),
        (712, 148),
        (535, 340),
        (720, 354),
        (568, 300),
        (629, 260),
        (539, 46),
        (634, 343),
        (491, 135),
        (768, 161),
    ],
    15: [
        (512, 317),
        (741, 72),
        (552, 50),
        (772, 346),
        (637, 12),
        (589, 131),
        (732, 165),
        (605, 15),
        (730, 38),
        (576, 216),
        (589, 381),
        (711, 387),
        (563, 228),
        (494, 22),
        (787, 288),
    ],
}


def generate_random_population(
    cities_location: list[tuple[float, float]], population_size: int
) -> list[list[tuple[float, float]]]:
    """
    Generate a random population of routes for a given set of cities.

    Parameters:
    - cities_location (list[tuple[float, float]]): A list of tuples representing the locations of cities,
      where each tuple contains the latitude and longitude.
    - population_size (int): The size of the population, i.e., the number of routes to generate.

    Returns:
    list[list[tuple[float, float]]]: A list of routes, where each route is represented as a list of city locations.
    """
    return [
        random.sample(cities_location, len(cities_location))
        for _ in range(population_size)
    ]


def calculate_distance(
    point1: tuple[float, float], point2: tuple[float, float]
) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - point1 (tuple[float, float]): The coordinates of the first point.
    - point2 (tuple[float, float]): The coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_fitness(path: list[tuple[float, float]]) -> float:
    """
    Calculate the fitness of a given path based on the total Euclidean distance.

    Parameters:
    - path (list[tuple[float, float]]): A list of tuples representing the path,
      where each tuple contains the coordinates of a point.

    Returns:
    float: The total Euclidean distance of the path.
    """
    distance = 0
    n = len(path)
    for i in range(n):
        distance += calculate_distance(path[i], path[(i + 1) % n])

    return distance


##################################
# using adjacency matrix


def generate_adjacency_matrix(
    cities_location: list[tuple[float, float]],
) -> np.ndarray:
    """
    Generate an adjacency matrix representing the distances between cities.

    Parameters:
    - cities_location (list[tuple[float, float]]): A list of tuples representing the locations of cities,
      where each tuple contains the latitude and longitude.

    Returns:
    np.ndarray: An adjacency matrix where the value at [i][j] is the Euclidean distance from city i to city j.
    """
    cities = np.array(cities_location)

    # Vectorized distance calculation - retorna apenas as distâncias
    distances = np.linalg.norm(cities[:, None] - cities, axis=2)

    return distances


def calculate_fitness_with_matrix_adjacency(
    path: list[tuple[float, float]],
    adjacency_matrix: np.ndarray,
    city_index_map: dict[tuple[float, float], int],
) -> float:
    """
    Calculate the fitness of a given path based on the total Euclidean distance using an adjacency matrix.

    Parameters:
    - path (list[tuple[float, float]]): A list of tuples representing the path.
    - adjacency_matrix (np.ndarray): The adjacency matrix with pre-calculated distances.
    - city_index_map (dict[tuple[float, float], int]): Dictionary mapping city coordinates to their indices.

    Returns:
    float: The total Euclidean distance of the path.
    """
    distance = 0
    n = len(path)
    for i in range(n):
        city1_index = city_index_map[path[i]]
        city2_index = city_index_map[path[(i + 1) % n]]

        distance += adjacency_matrix[city1_index, city2_index]

    return distance


#######################################


def order_crossover(
    parent1: list[tuple[float, float]], parent2: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    Perform order crossover (OX) between two parent sequences to create a child sequence.

    Parameters:
    - parent1 (list[tuple[float, float]]): The first parent sequence.
    - parent2 (list[tuple[float, float]]): The second parent sequence.

    Returns:
    list[tuple[float, float]]: The child sequence resulting from the order crossover.
    """
    length = len(parent1)

    # Choose two random indices for the crossover
    start_index = random.randint(0, length - 1)
    end_index = random.randint(start_index + 1, length)

    # Initialize the child with a copy of the substring from parent1
    child = parent1[start_index:end_index]

    # Fill in the remaining positions with genes from parent2
    remaining_positions = [
        i for i in range(length) if i < start_index or i >= end_index
    ]
    remaining_genes = [gene for gene in parent2 if gene not in child]

    for position, gene in zip(remaining_positions, remaining_genes):
        child.insert(position, gene)

    return child


### demonstration: crossover test code
# Example usage:
# parent1 = [(1, 1), (2, 2), (3, 3), (4,4), (5,5), (6, 6)]
# parent2 = [(6, 6), (5, 5), (4, 4), (3, 3),  (2, 2), (1, 1)]

# # parent1 = [1, 2, 3, 4, 5, 6]
# # parent2 = [6, 5, 4, 3, 2, 1]


# child = order_crossover(parent1, parent2)
# print("Parent 1:", [0, 1, 2, 3, 4, 5, 6, 7, 8])
# print("Parent 1:", parent1)
# print("Parent 2:", parent2)
# print("Child   :", child)


# # Example usage:
# population = generate_random_population(5, 10)

# print(calculate_fitness(population[0]))


# population = [(random.randint(0, 100), random.randint(0, 100))
#           for _ in range(3)]


# TODO: implement a mutation_intensity and invert pieces of code instead of just swamping two.
def mutate_shuffle_with_intensity(
    solution: list[tuple[float, float]],
    mutation_probability: float,
    mutation_intensity: float,
) -> list[tuple[float, float]]:
    """
    Mutate a solution by inverting a segment of the sequence with a given mutation probability.

    Parameters:
    - solution (list[tuple[float, float]]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation for each individual in the solution.
    - mutation_intensity (float): The intensity of the mutation, determining the length of the segment to be inverted.

    Returns:
    list[tuple[float, float]]: The mutated solution sequence.
    """
    mutated_solution = copy.deepcopy(solution)

    # Check if mutation should occur
    if random.random() < mutation_probability:

        # Ensure there are at least two cities to perform a swap
        if len(solution) < 2:
            return solution

        # Calculate segment size based on intensity
        # intensity 0.0 -> min segment (2 cities)
        # intensity 1.0 -> max segment (all cities)
        min_segment_size = 2
        max_segment_size = len(solution)
        segment_size = int(
            min_segment_size
            + mutation_intensity * (max_segment_size - min_segment_size)
        )

        # Ensure segment_size is at least 2 and doesn't exceed solution length
        segment_size = max(2, min(segment_size, len(solution)))

        # Select random starting index
        max_start_index = len(solution) - segment_size
        index1 = random.randint(0, max_start_index)
        index2 = index1 + segment_size - 1

        # Randomize the segment between the selected indices
        segment = mutated_solution[index1 : index2 + 1]
        random.shuffle(segment)
        mutated_solution[index1 : index2 + 1] = segment

    return mutated_solution


def mutate(
    solution: list[tuple[float, float]], mutation_probability: float
) -> list[tuple[float, float]]:
    """
    Mutate a solution by inverting a segment of the sequence with a given mutation probability.

    Parameters:
    - solution (list[tuple[float, float]]): The solution sequence to be mutated.
    - mutation_probability (float): The probability of mutation for each individual in the solution.

    Returns:
    list[tuple[float, float]]: The mutated solution sequence.
    """
    mutated_solution = copy.deepcopy(solution)

    # Check if mutation should occur
    if random.random() < mutation_probability:

        # Ensure there are at least two cities to perform a swap
        if len(solution) < 2:
            return solution

        # Select a random index (excluding the last index) for swapping
        index = random.randint(0, len(solution) - 2)

        # Swap the cities at the selected index and the next index
        mutated_solution[index], mutated_solution[index + 1] = (
            solution[index + 1],
            solution[index],
        )

    return mutated_solution


### Demonstration: mutation test code
# # Example usage:
# original_solution = [(1, 1), (2, 2), (3, 3), (4, 4)]
# mutation_probability = 1

# mutated_solution = mutate(original_solution, mutation_probability)
# print("Original Solution:", original_solution)
# print("Mutated Solution:", mutated_solution)


def sort_population(
    population: list[list[tuple[float, float]]], fitness: list[float]
) -> tuple[list[list[tuple[float, float]]], list[float]]:
    """
    Sort a population based on fitness values.

    Parameters:
    - population (list[list[tuple[float, float]]]): The population of solutions, where each solution is represented as a list.
    - fitness (list[float]): The corresponding fitness values for each solution in the population.

    Returns:
    tuple[list[list[tuple[float, float]]], list[float]]: A tuple containing the sorted population and corresponding sorted fitness values.
    """
    # Combine lists into pairs
    combined_lists = list(zip(population, fitness))

    # Sort based on the values of the fitness list
    sorted_combined_lists = sorted(combined_lists, key=lambda x: x[1])

    # Separate the sorted pairs back into individual lists
    sorted_population, sorted_fitness = zip(*sorted_combined_lists)

    return list(sorted_population), list(sorted_fitness)


if __name__ == "__main__":
    N_CITIES = 10

    POPULATION_SIZE = 100
    N_GENERATIONS = 100
    MUTATION_PROBABILITY = 0.3
    cities_locations = [
        (float(random.randint(0, 100)), float(random.randint(0, 100)))
        for _ in range(N_CITIES)
    ]

    # CREATE INITIAL POPULATION
    population = generate_random_population(cities_locations, POPULATION_SIZE)

    # Lists to store best fitness and generation for plotting
    best_fitness_values = []
    best_solutions = []

    for generation in range(N_GENERATIONS):

        population_fitness = [
            calculate_fitness(individual) for individual in population
        ]

        population, population_fitness = sort_population(population, population_fitness)

        best_fitness = calculate_fitness(population[0])
        best_solution = population[0]

        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        print(f"Generation {generation}: Best fitness = {best_fitness}")

        new_population = [population[0]]  # Keep the best individual: ELITISM

        while len(new_population) < POPULATION_SIZE:

            # SELECTION
            parent1, parent2 = random.choices(
                population[:10], k=2
            )  # Select parents from the top 10 individuals

            # CROSSOVER
            child1 = order_crossover(parent1, parent2)

            ## MUTATION
            child1 = mutate(child1, MUTATION_PROBABILITY)

            new_population.append(child1)

        print("generation: ", generation)
        population = new_population
