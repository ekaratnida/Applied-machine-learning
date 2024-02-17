
import math
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
#import pdb
#pdb.set_trace()

X, Y = make_classification(n_samples=10, n_features=4, random_state=42)
X =  X[:,0]
t1 = np.linspace(-80,80,num=100)

def sigmoid(x):
	result = 1 / (1 + math.exp(-x))
	return result 

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

plt.plot(t1,mse_sig(t1))
plt.show()