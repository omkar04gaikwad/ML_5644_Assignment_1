from sys import float_info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import csv
from csv_generator import x, labels
from variable import n, C, L, gmm_pdf, N, sample_path, plot_samples, scatter_plot, confusion, erm_classification

plot_samples(x, labels)

lambda1 = np.array([[0, 1, 1], [1, 0, 1], [10, 10, 0]])
decision_map1 = erm_classification(x, lambda1, gmm_pdf, C)
confusion_matrix2 = confusion(decision_map1, labels,'Confusion_Matrix1_Partb_1.png')
correct_class_samples2 = np.sum(np.diag(confusion_matrix2))

print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples2))
prob_error2 = 1 - (correct_class_samples2 / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error2))
print(x.shape)
scatter_plot(x,decision_map1, labels, L,name='ERM_Classification_scatter_plot_L10.png')

lambda2 = np.array([[0, 1, 1], [1, 0, 1], [100, 100, 0]])
decision_map2 = erm_classification(x, lambda2, gmm_pdf, C)
confusion_matrix3 = confusion(decision_map2, labels, 'Confusion_Matrix1_Partb_2.png')
correct_class_samples3 = np.sum(np.diag(confusion_matrix3))

print("Total Mumber of Misclassified Samples: {:d}".format(N - correct_class_samples3))
prob_error3 = 1 - (correct_class_samples3 / N)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error3))
scatter_plot(x,decision_map2, labels, L, name='ERM_Classification_scatter_plot_L100.png')