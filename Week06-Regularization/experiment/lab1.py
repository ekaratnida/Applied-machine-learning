import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
#X = np.array([[10, 15, 20, 25, 30]]).T
#y = np.array([[10, 30, 50, 51, 52]]).T

def cost_function(theta, x, y, m,lamda):    
    y_predict = theta.T.dot(x.T)
    print("y_predict ",y_predict.reshape(-1))
    print("y ",y.reshape(-1))
    y_predict = y_predict.reshape(-1)
    y = y.reshape(-1)
    diff = (y_predict-y)
    print("diff = ",diff)
    total=0
    for i in diff:
        total += i * i
    olsError =  total
    print("olsError ", olsError)
    l2Error = lamda * np.sum(np.abs(theta))
    error = (1/(2*m)) * (olsError + l2Error)
    print("error ", error)
    return error

def soft_threshold(rho,lamda):
    '''Soft threshold function used for normalized data and lasso regression'''
    if rho < - lamda:
        return (rho + lamda)
    elif rho >  lamda:
        return (rho - lamda)
    else:
        return 0

def gradient_descent(learning_rate, x, y, ep=0.0001, max_iter=10, lamda=0.1):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta
    #t = np.random.random((x.shape[1],1))
    t = np.ones((x.shape[1],1))
    #t = np.array([0.01,0.01,0.01,0.01,0.01]).reshape(5,1) 
    print("t ", t)
  
    # total error, J(theta)
    J = cost_function(t,x,y,m,lamda)
    #print("Iteration 0 --> J=",J," t0=",t0," t1=",t1)

    # Iterate Loop
    while not converged:

        y_predict = t.T.dot(x.T)
        print("y_predict ",y_predict)
        #print(type(y))        
        diff = y_predict-y
        print("diff ", diff)
        tmpAbs = [soft_threshold(i,lamda) for i in (t.T)[0]]
        print("tmpAbs ",tmpAbs)
        print("sum = ", np.sum(tmpAbs))
        lasso_error = lamda * np.sum(tmpAbs)
        print(lasso_error)
        print(type(x), " x shape = ", x.shape)
        print(type(diff), " diff shape = ", diff.shape)
        ols_error = diff.dot(x)
        print(ols_error)
        grad = (1/m) * (ols_error + lasso_error)

        print("grad = ",grad)

        #grad = x.T.dot(((t.T.dot(x.T)-y).T))

        t = t - learning_rate * (grad.T)
        print("t = ",t)
        
        # error
        e = cost_function(t,x,y,m,lamda)
        
        if abs(J-e) <= ep:
            print("Converged, iterations: ", iter, "/", max_iter)
            converged = True
    
        J = e   # update error s
        iter += 1  # update iter
    
        if iter == max_iter:
            print('Reaching Max iterations!')
            converged = True

        print(iter,"/",max_iter)
        
    return t


if __name__ == '__main__':
    
    learning_rate = 0.05 # learning rate
    poly_reg=PolynomialFeatures(degree=2)
    
    #df = pd.read_csv("http://freakonometrics.free.fr/chicago.txt",sep=";")
    df = pd.read_csv("data_hw1.txt",sep=",")
    print(df.head())
    
    #X_poly = poly_reg.fit_transform(X)
    
    x = df.loc[:, df.columns!='y']
    #x = x / (np.linalg.norm(x,axis = 0))
    x_b = np.c_[np.ones((x.shape[0],1)),x]
    print(x_b)
    y = df['y']
    print(y.head())
    y = y.to_numpy()
    theta_bgd = gradient_descent(learning_rate, x_b, y, max_iter=2,lamda=0)
    print ("theta ", theta_bgd)