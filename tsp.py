import pygame
import random
import itertools
from genetic_algorithm import (
    mutate,
    mutate_shuffle_with_intensity,
    order_crossover,
    generate_random_population,
    generate_adjacency_matrix,
    calculate_fitness_with_matrix_adjacency,
    calculate_fitness,
    sort_population,
    default_problems,
)
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import numpy as np
from benchmark_att48 import *


# Define constant values
# pygame
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450

STALE_LIMIT = 500  # Number of generations to wait for improvement before stopping

# GA
N_CITIES = 20
POPULATION_SIZE = 100
N_GENERATIONS = None
MUTATION_PROBABILITY = 0.5

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
secondary_path_colors = [
    (255, 200, 200),  # Rosa claro
    (200, 255, 200),  # Verde claro
    (255, 255, 180),  # Amarelo claro
    (220, 200, 255),  # Roxo claro
    (200, 240, 255),  # Ciano claro
]


# Initialize problem
# Using Random cities generation
cities_locations = [
    (
        float(random.randint(NODE_RADIUS + PLOT_X_OFFSET, WIDTH - NODE_RADIUS)),
        float(random.randint(NODE_RADIUS, HEIGHT - NODE_RADIUS)),
    )
    for _ in range(N_CITIES)
]


# # Using Deault Problems: 10, 12 or 15
# WIDTH, HEIGHT = 800, 400
# cities_locations = default_problems[15]


# Using att48 benchmark
# WIDTH, HEIGHT = 1500, 800
# att_cities_locations = np.array(att_48_cities_locations)
# max_x = max(point[0] for point in att_cities_locations)
# max_y = max(point[1] for point in att_cities_locations)
# scale_x = (WIDTH - PLOT_X_OFFSET - NODE_RADIUS) / max_x
# scale_y = HEIGHT / max_y
# cities_locations = [
#     (float(point[0] * scale_x + PLOT_X_OFFSET), float(point[1] * scale_y))
#     for point in att_cities_locations
# ]
# target_solution = [cities_locations[i - 1] for i in att_48_cities_order]
# fitness_target_solution = calculate_fitness(target_solution)
# print(f"Best Solution: {fitness_target_solution}")
# ----- Using att48 benchmark


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
start_time = pygame.time.get_ticks()
generation_counter = itertools.count(start=1)  # Start the counter at 1
stale_counter = 0


# Create Initial Population
# TODO:- use some heuristic like Nearest Neighbour our Convex Hull to initialize
population = generate_random_population(cities_locations, POPULATION_SIZE)

adjacency_matrix = generate_adjacency_matrix(cities_locations)
city_index_map = {city: i for i, city in enumerate(cities_locations)}

best_fitness_values = []
best_solutions = []
best_solution = None
algorithm_running = True  # Flag para controlar o algoritmo genético

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False
            elif event.key == pygame.K_c:
                algorithm_running = True
                stale_counter = 0
            elif event.key == pygame.K_p:
                algorithm_running = not algorithm_running

    if algorithm_running:
        generation = next(generation_counter)

        screen.fill(WHITE)

        cal_pop_start_time = pygame.time.get_ticks()
        # population_fitness = [
        #     calculate_fitness(individual) for individual in population
        # ]
        population_fitness = [
            calculate_fitness_with_matrix_adjacency(
                individual, adjacency_matrix, city_index_map
            )
            for individual in population
        ]
        cal_pop_end_time = pygame.time.get_ticks()
        print(
            f"Time to calculate fitness for population: {cal_pop_end_time - cal_pop_start_time} ms"
        )

        population, population_fitness = sort_population(population, population_fitness)

        best_fitness = population_fitness[0]
        best_solution = population[0]

        if best_fitness_values and best_fitness_values[-1] == best_fitness:
            stale_counter += 1
        best_fitness_values.append(best_fitness)
        best_solutions.append(best_solution)

        draw_plot(
            screen,
            list(range(len(best_fitness_values))),
            best_fitness_values,
            y_label="Fitness - Distance (pxls)",
        )

        draw_cities(screen, cities_locations, RED, NODE_RADIUS)
        draw_paths(screen, best_solution, BLUE, width=3)
        # Draw paths for individuals 1-5 with gradually different colors
        for i in range(1, 6):
            if i < len(population):
                color = secondary_path_colors[i - 1]
                draw_paths(screen, population[i], rgb_color=color, width=1)

        print(
            f"Generation {generation}: Best fitness = {round(best_fitness, 2)} - Time = {pygame.time.get_ticks() - start_time} ms"
        )

        # if target_solution == best_solution:
        #     print(
        #         f"Target solution found in generation {generation} with fitness {best_fitness}!"
        #     )
        #     break

        new_population = [population[0]]  # Keep the best individual: ELITISM
        # new_population = []  # No elitism

        while len(new_population) < POPULATION_SIZE:

            # selection
            # simple selection based on first 10 best solutions
            # parent1, parent2 = random.choices(population[:10], k=2)

            if stale_counter >= STALE_LIMIT:
                print(
                    f"No improvement in fitness for {STALE_LIMIT} generations. Applying tournament selection."
                )
                # tournament selection: select 10 random individuals and select the best among them as a parent
                tournament_size = 10
                tournament = random.sample(population[1:], tournament_size)
                parent1 = min(tournament, key=calculate_fitness)
                tournament = random.sample(population[1:], tournament_size)
                parent2 = min(tournament, key=calculate_fitness)
            else:
                # solution based on fitness probability
                probability = 1 / np.array(population_fitness)
                parent1, parent2 = random.choices(
                    population, weights=probability.tolist(), k=2
                )

            child1 = order_crossover(parent1, parent2)
            # child1 = order_crossover(parent1, parent1)
            if stale_counter >= STALE_LIMIT:
                print(
                    f"No improvement in fitness for {STALE_LIMIT} generations. Applying stronger mutation."
                )
                child1 = mutate_shuffle_with_intensity(child1, 1, 1)
            else:
                # child1 = mutate(child1, MUTATION_PROBABILITY)
                child1 = mutate_shuffle_with_intensity(child1, 0.7, 0.3)

            new_population.append(child1)

        population = new_population

        if stale_counter >= STALE_LIMIT:
            stale_counter = 0
        #     print(
        #         f"No improvement in fitness for {STALE_LIMIT} generations. Stopping early."
        #     )
        # algorithm_running = False
    else:
        screen.fill(WHITE)
        draw_plot(
            screen,
            list(range(len(best_fitness_values))),
            best_fitness_values,
            y_label="Fitness - Distance (pxls)",
        )
        draw_cities(screen, cities_locations, RED, NODE_RADIUS)
        draw_paths(screen, best_solution, BLUE, width=3)

    pygame.display.flip()
    clock.tick(FPS)

# TODO: save the best individual in a file if it is better than the one saved.
input("Press Enter to exit...")
# exit software
pygame.quit()
sys.exit()
