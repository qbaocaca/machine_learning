import numpy as np

def grad (x):
    return 4*np.cos(x) + 2*np.sin(2*x)

def cost (x):
    return 4*np.sin(x) - np.cos(2*x)

def myGD(eta, x0):
    x = [x0]
    for it in range(1000):
        x_new = x[-1] - eta *grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)

    return (x, it)

(x1, it1) = myGD(.1, -5)
(x2, it2) = myGD(.1, 5)

# print(x1)
# print(x2)

print(f'Solution x1= {x1[-1]}, '
      f'cost={cost(x1[-1])}, '
      f'obtained after {it1} iterations')

print(f'Solution x2= {x2[-1]}, '
      f'cost={cost(x2[-1])}, '
      f'obtained after {it2} iterations')