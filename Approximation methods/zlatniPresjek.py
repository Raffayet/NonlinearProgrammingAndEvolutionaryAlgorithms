import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin


def parabola(x1, x3, tol):
    
    X = np.array([x1, (x1+x3)/2, x3]).transpose()
    pom = np.array([1, 1, 1]).transpose()
    Y = np.array([pom, X, X*X]).transpose()

    F = np.linspace(0, 0, len(X))
    for i in range(0, len(X)):
        F[i] = func(X[i])

    abc = lin.solve(Y, F)

    x_opt = -abc[1]/(2*abc[2])
    f_opt = func(x_opt)
    n = 0

    while np.abs(np.dot([1, x_opt, x_opt*x_opt], abc) - f_opt) > tol:

        if x_opt > X[0] and x_opt < X[1]:
            if f_opt < F[0] and f_opt < F[1]:
                X = np.array([X[1], x_opt, X[2]])
                F = np.array([F[1], f_opt, F[2]])
            elif f_opt > F[1] and f_opt < F[0]:
                X = np.array([x_opt, X[1], X[2]])
                F = np.array([f_opt, F[1], F[2]])

            else:
                print("greska")

        elif x_opt > X[1] and x_opt < X[2]:
            if f_opt < F[1] and f_opt < F[2]:
                X = np.array([X[1], x_opt, X[2]])
                F = np.array([F[1], f_opt, F[2]])
            elif f_opt > F[1] and f_opt < F[2]:
                X = np.array([x_opt, X[1], X[2]])
                F = np.array([f_opt, F[1], F[2]])

            else:
                print("greska")

        else:
            print("x lezi van granica")

        pom = np.array([1, 1, 1]).transpose()
        Y = np.array([pom, X, X * X]).transpose()

        F = np.linspace(0, 0, len(X))
        for i in range(0, len(X)):
            F[i] = func(X[i])

        abc = lin.solve(Y, F)

        x_opt = -abc[1] / (2 * abc[2])
        f_opt = func(x_opt)
        n += 1

        return x_opt, f_opt, n



def func(x):
    f = -(x**4 - 5*x**3 - 2*x**2 + 24*x)
    return f

##################################################
# TESTIRANJE
a = 0
b = 2
tol = 0.001

[xopt, fopt, n] = parabola(a, b, tol)
print(xopt, fopt, n)

x = np.linspace(0, 4, 1000)
f = np.linspace(0, 0, len(x))
for i in range(0, len(x), 1):
    f[i] = func(x[i])
    
p = plt.plot(x, f, 'b--')
p = plt.plot(xopt, fopt, '*r', label = 'max[f(x)]', markersize = 15, markeredgewidth = 3)
plt.show()  

