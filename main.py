import matplotlib.pyplot as plt
import numpy as np

plt.close(fig='all')

# 1: Data
X = np.array([[-1], [1], [3], [5], [8], [9], [12]])
t = np.array([0, 0, 0, 1, 1, 1, 1])
f = np.zeros(7)

# 2: Algorithm
m = 1.5 # randomly initalize m
b = -1.5 # randomly initalize b

# 3: Train / Learn / Fit / Paramater Estimation
eta = 0.1 # learning rate
num_epochs = 420
Esaved = []
Msaved = []
Bsaved = []

for n in range(num_epochs):
    for i in range(len(X)): 
        y = 1/(1+np.exp(-(m*X[i, 0]+b))) # Computes the output for the algorithm
        e = ((y-t[i])**2)/2 # Finds the error of the output (E)
        Esaved.append(e)
        Msaved.append(m)
        Bsaved.append(b)
        
        pb = (y-t[i])*(y)*(1-y) # how much does E change when b changes (pE/pb)
        pm = pb*X[i, 0] # how much does E change when m changes (pE/pm)
        
        b = b - eta*(pb) # computes new value of b
        m = m - eta*(pm) # computes new value of m
        #print(m, b)

step = 0.5 # step of the x-axis for activation function and decision function
x_axis = np.arange(np.min(X), np.max(X)+step, step) # puts new data on the x-axis
deci_equ = m*x_axis+b # decision function
acti_equ = 1/(1+np.exp(-(m*x_axis+b))) # logistic sigmoid activation function (0 to 1)

plt.scatter(X, f, c=t, cmap='plasma')
plt.colorbar()
plt.plot(x_axis, deci_equ) # plotting decision function
plt.plot(x_axis, acti_equ) # plotting activation equation
plt.hlines(0.5, np.min(X), np.max(X), colors='C2') #plots a line at y=0.5
plt.ylim([-0.5, 1.5]) # limits graph to the range of logistic sigmoid function
plt.text(np.max(X)-2, 0.5, "threshold", c='C2')
plt.text(np.max(X)-2, 1, "activation", c='C1')
plt.text(np.mean(X), m*np.mean(X)+b, "decision", c='C0')
plt.grid()
