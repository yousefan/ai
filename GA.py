import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return -0.0001 * ((np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - (np.sqrt(x ** 2 + y ** 2) / np.pi)))) + 1) ** 0.1)


def cal_pop_fitness(pop):
    fitness = f(pop[:, 0], pop[:, 1])
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    parents = np.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.min(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = 99999999999
    return parents


def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1] / 2)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring


def mutation(offspring_crossover):
    for idx in range(offspring_crossover.shape[0]):
        random_value = np.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[idx, 1] = offspring_crossover[idx, 1] + random_value
    return offspring_crossover


equation_format = [4, -2]
num_params = len(equation_format)
sol_per_pop = 8
num_parents_mating = 4
pop_size = (sol_per_pop, num_params)
new_population = np.random.uniform(low=-4.0, high=4.0, size=pop_size)
print(new_population[:, 0])
print(new_population[:, 1])
print(f(new_population[:, 0], new_population[:, 1]))

best_out = []
generations = 10

for i in range(generations):
    fitness = cal_pop_fitness(new_population)
    best_out.append(np.min(f(new_population[:, 0], new_population[:, 1])))
    print("Generation:" + str(i) + "==> best output: " + str(np.min(f(new_population[:, 0], new_population[:, 1]))))
    parents = select_mating_pool(new_population, fitness, num_parents_mating)

    mCrossover = crossover(parents, offspring_size=(pop_size[0] - parents.shape[0], num_params))

    mMutation = mutation(mCrossover)

    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = mMutation

fitness = cal_pop_fitness(new_population)
best_match_idx = np.where(fitness == np.min(fitness))
answer = new_population[best_match_idx, :]
ansX = answer[0, 0, 0]
ansY = answer[0, 0, 1]
ansZ = fitness[best_match_idx]
ansZ = ansZ[0]
print("Best solution (X,Y): ", (ansX, ansY))
print("Best solution fitness : ", ansZ)

plt.plot(best_out)
plt.xlabel("Iteration")
plt.ylabel("Fitness")
plt.show()

x = np.linspace(-20, 20, 100)
y = np.linspace(-20, 20, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='viridis')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')