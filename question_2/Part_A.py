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



question_2_data = pd.read_csv(sample_path, delimiter=';')
question_2_data_length  = len(question_2_data.index)
question_2_data_array = question_2_data.iloc[:, :-1].to_numpy().T
label = question_2_data.iloc[:, -1].to_numpy()
N_per_l = np.array([sum(labels == l) for l in L])
print(N_per_l)
plot_samples(x, labels)


lambdas = np.ones((C, C)) - np.eye(C)
decision_map = erm_classification(x, lambdas, gmm_pdf, C)
confusion_matrix1 = confusion(decision_map, labels, 'Confusion_Matrix_PartA.png')

correct_class_samples1 = np.sum(np.diag(confusion_matrix1))
print("Total Mumber of Misclassified Samples:", "{:d}".format(N - correct_class_samples1))
prob_error1 = 1 - (correct_class_samples1 / N)
print("Empirically Estimated Probability of Error:", "{:.3f}".format(prob_error1))
print(L)
scatter_plot(x,decision_map, labels, L, 'ERM_Classification_scatter_plot.png')