from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

data = np.loadtxt('data.txt', delimiter=',')
#print(data[0:5])
X_train = data[:,[0,1]]
y_train = data[:,2]

plt.figure(figsize = (15,4), dpi=100)
plt.subplot(121)
plt.scatter(X_train[:,0],y_train)
plt.xlabel("Size of house (X1)")
plt.ylabel("Price (Y)")
plt.subplot(122)
plt.scatter(X_train[:,1],y_train)
plt.xlabel("Number of Bedrooms (X2)")
plt.ylabel("Price (Y)")

"""#Train model"""

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(lin_reg.intercept_)
print(lin_reg.coef_)

"""# Predict"""

X_test = np.array([[2000,6]])
result = lin_reg.predict(X_test)
print(result)

"""#Deploy"""

pickle.dump(lin_reg, open('model.sav','wb') )