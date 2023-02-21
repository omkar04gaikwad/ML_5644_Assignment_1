from urllib.request import urlopen  # library to open URLs
from zipfile import ZipFile  # library for working with ZIP files
from io import BytesIO  # library for working with binary data in memory
from sys import float_info  # provides information about the float type
import matplotlib.pyplot as plt  # library for creating visualizations
import numpy as np  # library for working with arrays and matrices
import pandas as pd  # library for working with data frames
from scipy.stats import norm, multivariate_normal  # statistical functions
from sklearn import preprocessing  # library for scaling and normalizing data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # library for creating confusion matrices
from sklearn.decomposition import PCA  # library for performing PCA
import csv  # library for reading and writing CSV files

# Downloads and reads in the HAR dataset without writing to disk
resp = urlopen('https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip')
har_zip = ZipFile(BytesIO(resp.read()))
har_train_df = pd.read_csv(har_zip.open('UCI HAR Dataset/train/X_train.txt'), delim_whitespace=True, header=None)
har_test_df = pd.read_csv(har_zip.open('UCI HAR Dataset/test/X_test.txt'), delim_whitespace=True, header=None)

# Concatenates the training and test data frames
har_df = pd.concat([har_train_df, har_test_df])

# Extracts the data matrix X
X = har_df.to_numpy()

# Creates a 3D plot to visualize the PCA
fig = plt.figure(figsize=(10, 10))
ax_pca = fig.add_subplot(111, projection='3d')

# Performs PCA with 3 principal components and fits the data
pca = PCA(n_components=3)
X_fit = pca.fit(X)
Z = pca.transform(X)

# Prints out the explained variance ratio of the PCA
print("Explained variance ratio: ", pca.explained_variance_ratio_)

# Plots the transformed data in 3D
ax_pca.scatter(Z[:, 0], Z[:, 1], Z[:, 2])
ax_pca.set_xlabel(r"$z_1$")
ax_pca.set_ylabel(r"$z_2$")
ax_pca.set_zlabel(r"$z_3$")
ax_pca.set_box_aspect((np.ptp(Z[:, 0]), np.ptp(Z[:, 1]), np.ptp(Z[:, 2])))

# Sets the title and displays the plot
plt.title("PCA of HAR Dataset")
plt.tight_layout()
plt.show()
