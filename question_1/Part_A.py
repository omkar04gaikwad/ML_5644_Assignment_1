from sys import float_info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import csv
from variable import m0,m1,C0,C1,N,priors,gmmParameters_q1, sample_path, class_length_q1, meanvectors_q1, covmatrices_q1, L_q1

def generate_scatter_plot(x, labels):
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[0, labels==0], x[1, labels==0], x[2, labels==0], c='r', alpha=1, label="Class 0")
    ax.scatter(x[0, labels==1], x[1, labels==1], x[2, labels==1], c='b', alpha=1, label="Class 1")
    ax.set_xlabel('X-scale')
    ax.set_ylabel('Y-scale')
    ax.set_zlabel('Z-scale')
    plt.title("Data and True Class Labels")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Samples_scatter_plot.png")
    plt.show()

def generate_roc(discriminant_score, labels):
    N_labels = np.array((sum(labels == 0), sum(labels == 1)))
    sorted_score = sorted(discriminant_score)
    gammas = ([sorted_score[0] - float_info.epsilon] +
              sorted_score +
              [sorted_score[-1] + float_info.epsilon])
    decisions = [discriminant_score >= g for g in gammas]
    ind10 = [np.argwhere((d == 1) & (labels == 0)) for d in decisions]
    p10 = [len(inds) / N_labels[0] for inds in ind10]
    ind11 = [np.argwhere((d == 1) & (labels == 1)) for d in decisions]
    p11 = [len(inds) / N_labels[1] for inds in ind11]
    roc = {'p10':np.array(p10),'p11':np.array(p11)}
    return roc, gammas

def roc_curve_generator(roc_erm):
    fig = plt.figure(figsize=(8, 8))
    roc = fig.add_subplot(111)
    roc.plot(roc_erm['p10'], roc_erm['p11'], color='r', label='ROC for ERM')
    plt.title("ROC Curve for ERM Classifier")
    roc.set_xlabel("Probability of False Positive")
    roc.set_ylabel("Probability of True Positive")
    roc.yaxis.grid(color='lightgrey', linestyle=':')
    roc.xaxis.grid(color='lightgrey', linestyle=':')
    roc.set_axisbelow(True)
    roc.set_xlim(0,1)
    roc.set_ylim(0,1)
    plt.tight_layout()
    plt.legend()
    plt.savefig("normal_roc_curve.png")
    plt.show()

def get_binary_classification_metrics(predictions, labels):
    N_labels = np.array((sum(labels == 0), sum(labels == 1)))
    ind_00 = np.argwhere((predictions == 0) & (labels == 0))
    ind_10 = np.argwhere((predictions == 1) & (labels == 0))
    ind_01 = np.argwhere((predictions == 0) & (labels == 1))
    ind_11 = np.argwhere((predictions == 1) & (labels == 1))
    metric_classes = {
        'TNR':len(ind_00) / N_labels[0],
        'FPR':len(ind_10) / N_labels[0],
        'FNR':len(ind_01) / N_labels[1],
        'TPR':len(ind_11) / N_labels[1]
    }
    return metric_classes
def generate_minimum_probab_roc(roc_erm):
    fig = plt.figure(figsize=(8, 8))
    roc = fig.add_subplot(111)
    roc.plot(roc_erm['p10'], roc_erm['p11'], color='r', label='ROC for ERM')
    roc.plot(roc_erm['p10'][empirical_integerof_minprobability],roc_erm['p11'][empirical_integerof_minprobability], 'go', label='Empirical Minimum Probability: {:.3f}'.format(empirical_minimum_probability_error), markersize=12)
    roc.plot(metric_classes['FPR'], metric_classes['TPR'], 'rx', label='Theoretical Minimum Probability: {:.3f}'.format(list_minimum_error_probab), markersize=12)
    plt.title("ROC Curve for ERM Classifier with minimum ERM classification error")
    roc.set_xlabel("Probability of False Positive")
    roc.set_ylabel("Probability of True Positive")
    roc.yaxis.grid(color='lightgrey', linestyle=':')
    roc.xaxis.grid(color='lightgrey', linestyle=':')
    roc.set_axisbelow(True)
    roc.set_xlim(0,1)
    roc.set_ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("minimum_roc_curve.png")
    plt.show()

question_1_data = pd.read_csv(sample_path, delimiter=';')
question_1_data_length  = len(question_1_data.index)
question_1_data_array = question_1_data.iloc[:, :-1].to_numpy().T
label = question_1_data.iloc[:, -1].to_numpy()
N_per_l_q1 = np.array([sum(label == l) for l in L_q1])
generate_scatter_plot(question_1_data_array, label)
likelihood_ratio_q1 = np.array([multivariate_normal.pdf(question_1_data_array.T, meanvectors_q1[p], covmatrices_q1[p]) for p in range(class_length_q1)])
discriminant_q1 = np.log(likelihood_ratio_q1[1]) - np.log(likelihood_ratio_q1[0])
roc_erm_q1, empirical_gamma_q1 = generate_roc(discriminant_q1, label)
roc_curve_generator(roc_erm_q1)

emprirical_error_probability = np.array((roc_erm_q1['p10'], 1 - roc_erm_q1['p11'])).T.dot(N_per_l_q1 / question_1_data_length)
empirical_minimum_probability_error = np.min(emprirical_error_probability)
empirical_integerof_minprobability = np.argmin(emprirical_error_probability)

gamma_val = priors[0] / priors[1]

decisions_map = discriminant_q1 >= np.log(gamma_val)

metric_classes = get_binary_classification_metrics(decisions_map, label)

list_minimum_error_probab = np.array((metric_classes['FPR'] * priors[0] + metric_classes['FNR'] * priors[1]))
generate_minimum_probab_roc(roc_erm_q1)


print("Theoretical Gamma: ", "{:.3f}".format(gamma_val))
print("Practical Gamma: ", "{:.3f}".format(np.exp(empirical_gamma_q1[empirical_integerof_minprobability])))
print("Minimum Theoretical probability: ", "{:.3f}".format(list_minimum_error_probab))
print("Minimum Emperical probability: ", "{:.3f}".format(empirical_minimum_probability_error))
