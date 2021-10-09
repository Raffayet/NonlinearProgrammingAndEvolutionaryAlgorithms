import numpy as np
from scipy.optimize import linprog

#y = -3x1 - 4x2 + 2x3 - 4x4 - x5
#x1 + x4 + x5 <= 30
#x2 + x3 + x5 <= 20
#x1 + x2 + x3 >= 10
#x2 + x4 = 5
#x3 + x5 = 15

c = np.array([-3, -4, 2, -4, -1])

A = np.array([[1, 0, 0, 1, 1],
              [0, 1, 1, 0, 1],
              [-1, -1, -1, 0, 0]])

b = np.array([30, 20, -10]).transpose()

Aeq = np.array([[0, 1, 0, 1, 0],
               [0, 0, 1, 0, 1]])

beq = np.array([5, 15]).transpose()

rezulat = linprog(c, A, b, Aeq, beq)
print(rezulat.x, -rezulat.fun)