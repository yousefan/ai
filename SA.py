import math
import random
import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return -0.0001 * (
        (np.abs(np.sin(x) * np.sin(y) * np.exp(np.abs(100 - (np.sqrt(x ** 2 + y ** 2) / np.pi)))) + 1) ** 0.1)


def random_gen():
    r = 5
    return random.random() * r - r / 2


def main():
    max_value = 10

    cuurent_x = random.randint(0, max_value)
    current_y = random.randint(0, max_value)

    temp = 100
    i = 1
    x_history = []
    y_history = []
    while 1:
        current_state = f(cuurent_x, current_y)

        temp_x = cuurent_x + random_gen()
        temp_y = current_y + random_gen()

        x_history.append(temp_x)
        y_history.append(temp_y)

        next_state = f(temp_x, temp_y)

        if next_state < current_state:
            current_y = temp_y
            cuurent_x = temp_x
        else:
            probablity = math.exp(-1 * (next_state - current_state) / temp)
            if random.random() <= probablity:
                current_y = temp_y
                cuurent_x = temp_x

        i = i + 1
        temp = temp * 0.95

        if i > 500:
            if current_state < next_state:
                next_state = current_state
            return x_history, y_history, cuurent_x, current_y, next_state


h_x, h_y, x, y, z = main()
print("desire value is (%f,%f,%f)" % (x, y, z))
plt.plot(h_x)
plt.plot(h_y)
plt.xlabel("Iteration")
plt.show()
