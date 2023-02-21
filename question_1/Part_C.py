from sys import float_info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from variable import m0,m1,C0,C1,N,priors,gmmParameters_q1, sample_path, class_length_q1, meanvectors_q1, covmatrices_q1, L_q1
from Part_A import roc_erm_q1 as roc_erm, empirical_integerof_minprobability, metric_classes, list_minimum_error_probab, empirical_minimum_probability_error
from Part_B import roc_naive_q1 as roc_naive, naive_minimum_interger_probability, naive_minimum_probability_error

def generate_data_from_gmm(N, pdf_params):
    n = pdf_params['mu'].shape[1]
    X = np.zeros([N, n])
    labels = np.zeros(N)
    u = np.random.rand(N)
    thresholds = np.cumsum(pdf_params['priors'])
    thresholds = np.insert(thresholds, 0, 0) 
    L = np.array(range(1, len(pdf_params['priors'])+1))
    for l in L:
        indices = np.argwhere((thresholds[l-1] <= u) & (u <= thresholds[l]))[:, 0]
        Nl = len(indices)  
        labels[indices] = l * np.ones(Nl) - 1
        if n == 1:
            X[indices, 0] =  norm.rvs(pdf_params['mu'][l-1], pdf_params['Sigma'][l-1], Nl)
        else:
            X[indices, :] =  multivariate_normal.rvs(pdf_params['mu'][l-1], pdf_params['Sigma'][l-1], Nl)
    
    return X, labels

def estimate_roc(discriminant_score, labels):
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
    roc = {}
    roc['p10'] = np.array(p10)
    roc['p11'] = np.array(p11)

    return roc, gammas

def perform_lda(X, labels, C=2):
    mu = np.array([np.mean(X[labels == i], axis=0).reshape(-1, 1) for i in range(C)])
    cov = np.array([np.cov(X[labels == i].T) for i in range(C)])
    Sb = (mu[1] - mu[0]).dot((mu[1] - mu[0]).T)
    Sw = cov[0] + cov[1]
    lambdas, U = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))

    idx = lambdas.argsort()[::-1]

    U = U[:, idx]

    w = U[:, 0]

    z = X.dot(w)

    return w, z

N = 10000
gmm_pdf = {}
gmm_pdf['priors'] = np.array([0.35, 0.65])
num_classes = len(gmm_pdf['priors'])
gmm_pdf['mu'] = np.array([[-0.5, -0.5, -0.5, -0.5], [1, 1, 1, 1]])  
gmm_pdf['Sigma'] =np.array([[[2/4, -0.5/4, 0.3/4, 0], [-0.5/4, 1/4, -0.5/4, 0], [0.3/4, -0.5/4, 1/4, 0], [0, 0, 0, 2/4]],[[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]])  # Gaussian distributions covariance matrices

X, labels = generate_data_from_gmm(N, gmm_pdf)
n = X.shape[1]
L = np.array(range(num_classes))
N_per_l = np.array([sum(labels == l) for l in L])
_, discriminant_score_lda = perform_lda(X, labels)

roc_lda, gamma_lda = estimate_roc(discriminant_score_lda, labels)

prob_error_lda = np.array((roc_lda['p10'], (1 - roc_lda['p11']))).T.dot(N_per_l / N)


min_prob_error_lda = np.min(prob_error_lda)
min_ind_lda = np.argmin(prob_error_lda)

fig = plt.figure(figsize=(8, 8))
roc = fig.add_subplot(111)
roc.plot(roc_erm['p10'], roc_erm['p11'], color='r', label="ROC of ERM")
roc.plot(roc_naive['p10'], roc_naive['p11'], color='g', label="ROC of Naive")
roc.plot(roc_erm['p10'][empirical_integerof_minprobability],roc_erm['p11'][empirical_integerof_minprobability], 'go', label='Empirical Minimum Probability: {:.3f}'.format(empirical_minimum_probability_error), markersize=12)
roc.plot(metric_classes['FPR'], metric_classes['TPR'], 'rx', label='Theoretical Minimum Probability: {:.3f}'.format(list_minimum_error_probab), markersize=12)
roc.plot(roc_naive['p10'][naive_minimum_interger_probability], roc_naive['p11'][naive_minimum_interger_probability], 'b+', label='Minimum Naive Probability: {:.3f}'.format(naive_minimum_probability_error),
        markersize=12)
roc.plot(roc_lda['p10'], roc_lda['p11'], label="ROC of LDA")
roc.plot(roc_lda['p10'][min_ind_lda], roc_lda['p11'][min_ind_lda], 'm*', label="Minimum Error Probability for LDA: pyt{:.3f}".format(min_prob_error_lda),  markersize=16)
roc.yaxis.grid(color='lightgrey', linestyle=':')
roc.xaxis.grid(color='lightgrey', linestyle=':')
roc.set_axisbelow(True)
roc.set_xlim(0,1)
roc.set_ylim(0,1)
plt.legend()
plt.tight_layout()
plt.savefig("LDA_roc_curve.png")
plt.show()

print("Minimum Error Probability for LDA: ", "{:.3f}".format(min_prob_error_lda))
print("Minimum Gamma for LDA: ",  "{:.3f}".format(np.real(gamma_lda[min_ind_lda])))