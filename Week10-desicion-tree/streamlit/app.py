import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import random

x, y = make_blobs(centers=2,cluster_std=10, random_state=123)
#inx =  random.randint(0,len(x))

st.write(x.shape)
test = x[len(x)-1,].reshape(1,-1)
st.write(test)

num_of_neighbors = st.slider("num of neighbors",1,30)
neigh = NearestNeighbors(n_neighbors=num_of_neighbors)
neigh.fit(x[:-1,:], y[:-1])

result = neigh.kneighbors(test)
#print("result = ",result) #distance and index

#Plot all points
fig, ax = plt.subplots(figsize=(12,8))
plt.scatter(x[:, 0], x[:, 1], c = y, s=50, cmap='viridis')

#Plot neighbors
plt.scatter(x[result[1][0:], 0], x[result[1][0:], 1], c='red', s=100, alpha=0.7)

#Plot target
plt.scatter(test[0,0], test[0,1], c='green', s=100, alpha=1)

#plt.show()
st.pyplot(fig)
