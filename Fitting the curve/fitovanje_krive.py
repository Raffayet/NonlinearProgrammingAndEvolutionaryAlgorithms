import numpy as np
import matplotlib.pyplot as plt

file = open("data")

content = file.readlines()

x_coordinates = []
y_coordinates = []

for line in content:
    words = line.split(",")
    x_coordinates.append(eval(words[0]))
    y_coordinates.append(eval(words[1]))

file.close()

n = len(x_coordinates)
sum_of_x = 0
sum_of_y = 0
sum_of_xy = 0
sum_of_xx = 0

for i in range(0, n):
    sum_of_x += x_coordinates[i]
    sum_of_y += y_coordinates[i]
    sum_of_xy += x_coordinates[i]*y_coordinates[i]
    sum_of_xx += x_coordinates[i]*x_coordinates[i]

a1 = (n*sum_of_xy - sum_of_x*sum_of_y)/(n*sum_of_xx - sum_of_x*sum_of_x)
a0 = sum_of_y/n - a1*sum_of_x/n

print(a1)
print(a0)

x = np.linspace(0, 250, 1000)
y = a1*x + a0


def function(a0, a1, x_vector):
    return a1*x_vector[i] + a0


def output(x_coordinates, y_coordinates, x, y):

    for i in range(0, len(x_coordinates)):

        p = plt.plot(x_coordinates[i], y_coordinates[i], "og", markersize=1, markeredgewidth=1)

    p = plt.plot(x, y, 'r')
    plt.show()


output(x_coordinates, y_coordinates, x, y)


#for i in range(0, len(x_vector), 1):
 #   y_vector[i] = function(a0, a1, x_vector)
