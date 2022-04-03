from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
 
st.title('Iris')
 
 
df = pd.read_csv("https://raw.githubusercontent.com/ekaratnida/Applied-machine-learning/a141c5d86dd0339be4ccbc9f82a8716251ffd6f3/Week10-desicion-tree/iris.csv")
 
if st.checkbox('Show dataframe'):
    st.write(df)
 
st.subheader('Scatter plot')
 
species = st.multiselect('Show iris per variety?', df['Species'].unique())
col1 = st.selectbox('Which feature on x?', df.columns[1:5])
col2 = st.selectbox('Which feature on y?', df.columns[1:5])
 
new_df = df[(df['Species'].isin(species))]
st.write(new_df)
# create figure using plotly express
fig = px.scatter(new_df, x =col1,y=col2, color='Species')
# Plot!
 
 
st.plotly_chart(fig)
 
st.subheader('Histogram')
 
feature = st.selectbox('Which feature?', df.columns[1:5])
# Filter dataframe
new_df2 = df[(df['Species'].isin(species))][feature]
fig2 = px.histogram(new_df, x=feature, color="Species", marginal="rug")
st.plotly_chart(fig2)
 
st.subheader('Machine Learning models')
 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
 
 
features= df[['SepalLength[cm]', 'SepalWidth[cm]', 'PetalLength[cm]', 'PetalWidth[cm]']].values
labels = df['Species'].values
 
X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
 
alg = ['Decision Tree', 'Logistic Regression', 'K-NN', 'Support Vector Machine']
classifier = st.selectbox('Which algorithm?', alg)
if classifier=='Decision Tree':
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    acc = dtc.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_dtc = dtc.predict(X_test)
    cm_dtc=confusion_matrix(y_test,pred_dtc)
    st.write('Confusion matrix: ', cm_dtc)

elif classifier == 'Logistic Regression':
    logist = LogisticRegression()
    logist.fit(X_train, y_train)
    acc = logist.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = logist.predict(X_test)
    cm=confusion_matrix(y_test,pred_svm)
    st.write('Confusion matrix: ', cm)

elif classifier == 'K-NN':
    knn=KNeighborsClassifier()
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = knn.predict(X_test)
    cm=confusion_matrix(y_test,pred_svm)
    st.write('Confusion matrix: ', cm)
 
elif classifier == 'Support Vector Machine':
    svm=SVC()
    svm.fit(X_train, y_train)
    acc = svm.score(X_test, y_test)
    st.write('Accuracy: ', acc)
    pred_svm = svm.predict(X_test)
    cm=confusion_matrix(y_test,pred_svm)
    st.write('Confusion matrix: ', cm)