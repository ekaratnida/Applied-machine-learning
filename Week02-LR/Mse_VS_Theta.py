
import math
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X,Y = make_regression(n_samples=10, n_features=1, noise=1, random_state=42)
t1 = np.linspace(-80,80,num=100)

def mse(t1):
	J=[]
	for t in t1:
		yh = 1 + np.multiply(t,X)
		sq_er = (yh-Y)**2
		asqe = np.sum(sq_er)/len(Y)
		J.append(asqe)
	return J

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def mse_sig(t1):
	J=[]
	for t in t1:
		yh = 1 + np.multiply(t,X)
		yhs = []
		for i in yh:
			yhs.append(sigmoid(i))
		yh = yhs 
		sq_er = (yh-Y)**2
		asqe = np.sum(sq_er)/len(Y)
		J.append(asqe)
	return J


#plt.plot(t1,mse_sig(t1))
plt.plot(t1,mse(t1))
plt.show()