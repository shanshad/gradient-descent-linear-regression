import numpy as np
import matplotlib.pyplot as plt
X = np.array([
    [1, 0, 2, 1, 3],
    [2, 1, 0, 2, 4],
    [3, 1, 1, 0, 2],
    [4, 2, 1, 1, 5],
    [5, 3, 2, 1, 1],
    [6, 2, 3, 2, 4],
    [7, 4, 2, 3, 3],
    [8, 5, 3, 2, 6]
])
y = np.array([[10], [12], [11], [20], [18], [25], [28], [35]])

m=X.shape[0]
Xn=np.hstack((np.ones((m,1)),X))
thetas=np.zeros((Xn.shape[1],1))

def Yhat(Xn,thetas):
    return (Xn@thetas)

def Cost(Xn,thetas,y):
    yhat=Yhat(Xn,thetas)
    m=yhat.shape[0]
    cost=(1/2*m)*((yhat-y).T@(yhat-y))
    return cost

def gradient(Xn,thetas,y):
    yhat=Yhat(Xn,thetas)
    m=yhat.shape[0]
    grad=(Xn.T@(yhat-y))/m
    return grad

alpha=0.01
prev_cost=Cost(Xn,thetas,y)
cost_history=[]
s=0
for i in range(1000):
    thetas=thetas-alpha*gradient(Xn,thetas,y)
    cost=Cost(Xn,thetas,y)
    cost_history.append(cost)
    if abs(cost.item()-prev_cost.item())<0.0001:
        s=i
        print(cost,s)
        break
    prev_cost=cost
cst_lst=[a.item() for a in cost_history]
x_range=range(s)
print(cst_lst)
plt.plot(cst_lst)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs Iteration")
plt.show()
