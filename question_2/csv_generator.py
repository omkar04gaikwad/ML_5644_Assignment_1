from sys import float_info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import csv
from variable import N,gmm_pdf ,sample_path


def generateDataFromGMM(N,pdf, filename):
    n= pdf['meanvectors'].shape[1]
    x = np.zeros([N,n])
    labels = np.zeros(N)
    u = np.random.rand(N)
    thresholds = np.cumsum(pdf['priors'])
    thresholds = np.insert(thresholds, 0, 0)
    L = np.array(range(1, len(pdf['priors'])+1))
    for l in L:
        indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
        N1 = len(indices)
        labels[indices] = l * np.ones(N1) - 1
        if n == 1:
            x[indices, 0] = norm.rvs(pdf['meanvectors'][l-1], pdf['covmatrices'][l-1], N1)
        else:
            x[indices, :] = multivariate_normal.rvs(pdf['meanvectors'][l-1], pdf['covmatrices'][l-1], N1)
    with open(filename, mode='w') as file:
      writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['x', 'y', 'z', 't', 'labels'])
      for i in range(N):
        row = list(x.T[:, i]) + [labels[i]]
        writer.writerow(row)
    return x, labels

x, labels = generateDataFromGMM(N,gmm_pdf ,sample_path)
print("sample saved successfully")

