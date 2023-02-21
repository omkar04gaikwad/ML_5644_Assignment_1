from sys import float_info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import csv

n = 3
C = 3
gmm_pdf = {
    'priors': np.array([0.3, 0.3, 0.4]),
    'meanvectors': np.array([1*np.ones(n), 2*np.ones(n), 3*np.ones(n)]),
    'covmatrices': np.array([3*np.eye(n), 3*np.eye(n), 3*np.eye(n)])
}
N = 10000
L = np.array(range(C))
lambda1 = np.array([[0, 1, 1], [1, 0, 1], [10, 10, 0]])
lambda2 = np.array([[0, 1, 1], [1, 0, 1], [100, 100, 0]])
sample_path = 'C:\\Users\\omkar\\OneDrive\\Desktop\\eece5644\\HW1\\question_2\\sample_list.csv'

def plot_samples(x, labels):
  fig = plt.figure(figsize = (9,9))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x[labels==0, 0], x[labels==0, 1], x[labels==0, 2], 'r.', alpha=1, label="Class 0")
  ax.scatter(x[labels==1, 0], x[labels==1, 1], x[labels==1, 2], 'r.', alpha=1, label="Class 1")
  ax.scatter(x[labels==2, 0], x[labels==2, 1], x[labels==2, 2], 'r.', alpha=1, label="Class 2")
  ax.set_xlabel('X-scale')
  ax.set_ylabel('Y-scale')
  ax.set_zlabel('Z-scale')
  plt.title("Data and True Class Labels")
  plt.legend()
  plt.tight_layout()
  ax.legend(loc='upper left', title='Class Label')
  plt.savefig('sample_scatter.png')
  plt.show()

def erm_classification(x, lambdas, gmm_pdf, C):
  likelihood_ratio_class = np.array([multivariate_normal.pdf(x, gmm_pdf['meanvectors'][i], gmm_pdf['covmatrices'][i]) for i in range(C)])
  Class_priors = np.diag(gmm_pdf['priors'])
  class_posteriors = Class_priors.dot(likelihood_ratio_class)
  riskmatrix = lambdas.dot(class_posteriors)
  return np.argmin(riskmatrix, axis=0)

def confusion(decision_map, labels, name):
  print("Confusion Matrix rows: Predicted class, columns: True class")
  confusion_matrix1 = confusion_matrix(decision_map, labels)
  conf_display = ConfusionMatrixDisplay.from_predictions(decision_map, labels, display_labels=['0', '1', '2'], colorbar=True)
  plt.ylabel("Predicted Labels")
  plt.xlabel("True Labels")
  plt.savefig(name)
  plt.show()
  return confusion_matrix1


def scatter_plot(x, decision_map, labels, L,name):
  figmap = plt.figure(figsize=(9,9))
  ax = figmap.add_subplot(111, projection='3d')
  markers = 'o^s'
  for r in L:
    for c in L:
      index_roc = np.argwhere((decision_map == r) & (labels == c))
      if r == c:
        marks = markers[r] + 'g'
        ax.plot(x[index_roc, 0].flatten(), x[index_roc, 1].flatten(), marks,label="D = {} | L = {}".format(r, c))
      else:
        marks = markers[r] + 'r'
        ax.plot(x[index_roc, 0].flatten(), x[index_roc, 1].flatten(), marks,label="D = {} | L = {}".format(r, c))
  plt.legend()
  plt.xlabel(r"$x_1$")
  plt.ylabel(r"$x_2$")
  plt.title("Classification Decisions: Predictions,True Labels")
  plt.tight_layout()
  plt.savefig(name)
  plt.show()