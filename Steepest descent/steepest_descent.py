import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def steepest_descent(gradf, x0, gamma, omega1, omega2, tol, epsilon, iterations):
    x = [np.array(x0).reshape(len(x0), 1)]
    v = [np.ones(x[-1].shape)]
    m = [np.ones(x[-1].shape)]
    for i in range(iterations):
        g = np.asarray(gradf(x[-1]))
        v.append(v[-1]*omega2 + (1 - omega2)*np.multiply(g, g))
        m.append(m[-1]*omega1 + (1 - omega1)*g)
        hat_v = np.abs(v[-1]/(1 - omega2))
        hat_m = m[-1]/(1 - omega1)
        x.append(x[-1] - gamma*np.ones(g.shape)*hat_m/np.sqrt(hat_v + epsilon))
        if np.linalg.norm(g) < tol:
            break

    return x[-1], v, m


# f = x1^2 + x1x2 + 0.5x2^2 + x1 + 10x2
def func(x):
    f = x[0]*x[0] + x[0]*x[1] + 0.5*x[1]*x[1] + x[0] + 10*x[1]
    return f


def gradf(x):
    df = np.zeros((2, 1))
    df[0] = 2*x[0] + x[1] + 1
    df[1] = x[0] + x[1] + 10
    return df

x0=[3, 0.1]
gamma=0.091
omega1=0.9
omega2=0.99
epsilon1=1e-6
epsilon=1e-6

N=100
xopt, v, m = adam(gradf, x0, gamma, omega1, omega2, epsilon, epsilon1, N)
fopt = func(xopt)
print("Funkcija ima minimum fopt = " , fopt , "u tacki: " , xopt ,".\n")

x1v = np.arange(-5, 15, 0.1)
x2v = np.arange(-5, 15, 0.1)
x1, x2 = np.meshgrid(x1v, x2v)
m, n = np.shape(x1)
f = np.zeros((m, n))
for i in range(0, m):
    for j in range(0, n):
        f[i][j] = func([x1[i][j], x2[i][j]])


fig = plt.figure()
ax = fig.gca(projection='3d')
p1 = ax.plot_surface(x1, x2, f)
ax.scatter(xopt[0],xopt[1], func(xopt), c='r', marker='o')
plt.show()
