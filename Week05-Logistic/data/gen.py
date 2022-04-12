import numpy as np

isMSE = True

t0 = np.linspace(-5.0, 5.0, num=101)
#print(len(t0))

t1 = np.linspace(-5.0, 5.0, num=101)
#print(t1)

data = {(2,0),(3,0),(4,1),(8,1)}
size =  test = data.__len__()  
errs = []
for _t0 in t0:
    for _t1 in t1:
        err_list = []
        for d in data:
            y_pred = 1/(1 + np.exp(-(_t0 + _t1*d[0])))
            if y_pred == 1:
                y_pred = 0.9999999999

            if isMSE:
                # Square error
                error = (y_pred - d[1])*(y_pred - d[1])

            else:
            # Negative log loss
                error = (d[1]*np.log(y_pred))+((1-d[1])*np.log(1-y_pred))
            
            err_list.append(error)
        
        if isMSE:
            e = (np.sum(err_list)/size)
        else:
            e = (np.sum(err_list)/size)*-1

        errs.append(str(_t0)+" "+" "+str(_t1)+" "+str(e))
        #print(errs)

outputFile = ''

if isMSE:
    outputFile = 'mse.txt'
else:
    outputFile = 'nll.txt'

with open(outputFile, 'w') as f:
    for item in errs:
        f.write("%s\n" % item)