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
from Part_A import roc_erm_q1, empirical_integerof_minprobability, metric_classes, list_minimum_error_probab, empirical_minimum_probability_error

question_1_data = pd.read_csv(sample_path, delimiter=';')
question_1_data_length  = len(question_1_data.index)
question_1_data_array = question_1_data.iloc[:, :-1].to_numpy().T
label = question_1_data.iloc[:, -1].to_numpy()
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

def naive_roc_generator(roc_erm,roc_naive):
    fig = plt.figure(figsize=(8, 8))
    roc = fig.add_subplot(111)
    roc.plot(roc_erm['p10'], roc_erm['p11'], color='r', label="ROC of ERM")
    roc.plot(roc_naive['p10'], roc_naive['p11'], color='g', label="ROC of Naive")
    roc.plot(roc_erm['p10'][empirical_integerof_minprobability],roc_erm['p11'][empirical_integerof_minprobability], 'go', label='Empirical Minimum Probability: {:.3f}'.format(empirical_minimum_probability_error), markersize=12)
    roc.plot(metric_classes['FPR'], metric_classes['TPR'], 'rx', label='Theoretical Minimum Probability: {:.3f}'.format(list_minimum_error_probab), markersize=12)
    roc.plot(roc_naive['p10'][naive_minimum_interger_probability], roc_naive['p11'][naive_minimum_interger_probability], 'b+', label='Minimum Naive Probability: {:.3f}'.format(naive_minimum_probability_error),
            markersize=12)
    plt.title("ROC Curve for NaiveBayesian Classifier")
    roc.set_xlabel("Probability of False Positive")
    roc.set_ylabel("Probability of True Positive")
    roc.yaxis.grid(color='lightgrey', linestyle=':')
    roc.xaxis.grid(color='lightgrey', linestyle=':')
    roc.set_axisbelow(True)
    roc.set_xlim(0,1)
    roc.set_ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Naive_roc_curve.png")
    plt.show()

n = question_1_data_array.T.shape[1]
Naive_likelihood = np.array([multivariate_normal.pdf(question_1_data_array.T,gmmParameters_q1['meanVectors'].T[l], np.eye(n)) for l in L_q1])
Naive_discriminant = np.log(Naive_likelihood[1]) - np.log(Naive_likelihood[0])

roc_naive_q1, gammas_naive_q1 = generate_roc(Naive_discriminant, label)
N_per_l_q1 = np.array([sum(label == l) for l in L_q1])
Naive_error_probability = np.array((roc_naive_q1['p10'], (1 - roc_naive_q1['p11']))).T.dot(N_per_l_q1 / N)

naive_minimum_probability_error = np.min(Naive_error_probability)
naive_minimum_interger_probability = np.argmin(Naive_error_probability)
naive_roc_generator(roc_erm_q1,roc_naive_q1)

print("Naive Gamma: ", "{:.3f}".format(np.exp(gammas_naive_q1[naive_minimum_interger_probability])))
print("Minimum Theoretical probability: ", "{:.3f}".format(list_minimum_error_probab))
print("Naive Emperical Probability: ",  "{:.3f}".format(naive_minimum_probability_error))