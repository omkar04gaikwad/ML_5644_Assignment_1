from sys import float_info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import csv
from variable import m0,m1,C0,C1,N,priors,gmmParameters_q1, sample_path


def generateDataFromGMM(N, gmmParameters, filename):
    priors = gmmParameters['priors']
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]
    C = len(priors)
    x = np.zeros((n, N))
    labels = np.zeros(N, dtype=int)
    u = np.random.rand(N)
    thresholds = np.concatenate(([0], np.cumsum(priors)))
    for l in range(C):
        indl = np.where(u <= thresholds[l+1])[0]
        Nl = len(indl)
        labels[indl] = l * np.ones(Nl, dtype=int)
        u[indl] = 1.1 * np.ones(Nl)
        x[:, indl] = np.random.multivariate_normal(meanVectors[:, l], covMatrices[:, :, l], Nl).T
    with open(filename, mode='w') as file:
      writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
      writer.writerow(['x', 'y', 'z', 't', 'labels'])
      for i in range(N):
        row = list(x[:, i]) + [labels[i]]
        writer.writerow(row)
    return "sample saved successfully"

status = generateDataFromGMM(N, gmmParameters_q1, sample_path)
print(status)

