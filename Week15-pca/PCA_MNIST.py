# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 15:06:36 2020

@author: Sumet Ketsri
"""

#-----------------------------------------------------------------------------
# %% Load mnist_784 dataset
from sklearn.datasets import fetch_openml
import numpy as np
from matplotlib import pyplot as plt
mnist = fetch_openml('mnist_784')
print()

#-----------------------------------------------------------------------------
# %% Split train, test data and save original test data
from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, 
                                                            mnist.target, 
                                                            test_size=0.2, # test_size=0.2 means train=80%, test=20%
                                                            random_state=0)
test_img_original = test_img.copy() # save original test images for plot
print(f"Train data has {train_img.shape[0]} images, each image has {train_img.shape[1]} pixels.")
print(f"Test data has {test_img.shape[0]} images, each image has {test_img.shape[1]} pixels.")
print()

#-----------------------------------------------------------------------------
# %% Preprocessing
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_img)
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#-----------------------------------------------------------------------------
# %% Do Principle Components Analysis
# and transform the input (prepare for modeling in the next step)
from sklearn.decomposition import PCA
print(f"Before PCA, there are {train_img.shape[1]} components.")
beta = 0.5
pca = PCA(beta)
pca.fit(train_img)
print(f"After PCA with {beta*100}% of variance, there are {pca.n_components_} components remain.") # 327 components remain
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
print()

#-----------------------------------------------------------------------------
# %% Build model using Logistic Regression
from sklearn.linear_model import LogisticRegression
print("Training model using Logistic Regression, please wait...")
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=1000)
logisticRegr.fit(train_img, train_lbl)
print()

#-----------------------------------------------------------------------------
# %% Predict for the test data and display some results

#[Method 1: Display as individual result]
# max_index = 30
# for i in range(max_index):
#     # Convert array size 784x1 to 28x28 to be able to plot
#     my_image = np.array(test_img_original[i], dtype='float')
#     pixels = my_image.reshape((28, 28))

#     # Plot image with color mapping (cmap) in grayscale
#     plt.imshow(pixels, cmap='gray')
#     plt.show()
    
#     # Print the predicted result
#     result = logisticRegr.predict(test_img[i].reshape(1,-1))
#     print(f"Predicted as {result[0]}")

# [Method 2: Display in subplot]
n_rows = 5
n_cols = 10
max_index = n_rows*n_cols
fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols, n_rows),
                        subplot_kw={'xticks': [], 'yticks': []})

print("Note that the title of each image is the predicted result.")
for ax, i in zip(axs.flat, range(max_index)):
    # Convert array size 784x1 to 28x28 to be able to plot
    my_image = np.array(test_img_original[i], dtype='float')
    pixels = my_image.reshape((28, 28))
    
    # Plot image with color mapping (cmap) in grayscale
    ax.imshow(pixels, cmap='gray')
    # Set title with its result
    result = logisticRegr.predict(test_img[i].reshape(1,-1)) # convert dimension (327,) to (1,327)
    ax.set_title(result[0])

# Display plot
plt.tight_layout()
plt.show()
