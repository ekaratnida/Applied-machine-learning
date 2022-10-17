import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

x, y = make_blobs(n_samples = 20, centers=2,cluster_std=3, random_state=123)
#inx =  random.randint(0,len(x))

st.write(x.shape)

target = x[x.shape[0]-1,].reshape(1,-1)

def getTargetPoint():
    rp = np.random.randint(1,x.shape[0])
    target = x[rp,].reshape(1,-1)
    st.write(target)
    return target

if st.button('Random test point'):
    target = getTargetPoint()

num_of_neighbors = st.slider("num of neighbors",1,x.shape[0])
neigh = NearestNeighbors(n_neighbors = num_of_neighbors)
neigh.fit(x[:-1,:], y[:-1])

result = neigh.kneighbors(target)
#print("result = ",result) #distance and index

#Plot all points
fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(x[:, 0], x[:, 1], c = y, s=50, cmap='viridis')

#Plot neighbors
plt.scatter(x[result[1][0:], 0], x[result[1][0:], 1], c='blue', s=200, alpha=1)

#Plot target
plt.scatter(target[0,0], target[0,1], c='red', s=200, alpha=1)

#plt.show()
st.pyplot(fig)
