from sys import float_info
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import csv

def covariance_regularization(sample, lambda_reg):
  n = sample.shape[0]
  covmatrix = np.cov(sample)
  covmatrix += lambda_reg * np.eye(n)
  return covmatrix

def erm_classification1(x, lambdas, gmm_pdf, C):
  likelihood_ratio_class = np.array([multivariate_normal.pdf(x, gmm_pdf['meanvectors'][i], gmm_pdf['covmatrices'][i]) for i in range(C)])
  Class_priors = np.diag(gmm_pdf['priors'])
  class_posteriors = Class_priors.dot(likelihood_ratio_class)
  riskmatrix = lambdas.dot(class_posteriors)
  return np.argmin(riskmatrix, axis=0)


def confusion1(decision_map, labels):
  print("Confusion Matrix rows: Predicted class, columns: True class")
  confusion_matrix1 = confusion_matrix(decision_map, labels)
  fig, ax = plt.subplots(figsize=(10, 10))
  conf_display = ConfusionMatrixDisplay.from_predictions(decision_map, labels, ax=ax, display_labels=['3', '4', '5', '6', '7', '8', '9'], colorbar=True)
  plt.ylabel('Predicted Labels')
  plt.xlabel('True Labels')
  return confusion_matrix1


def scatter_plot_wine(wine_df):
  fig = plt.figure(figsize=(10, 10))
  ax_subset = fig.add_subplot(111, projection='3d')
  unique_qualities = np.sort(wine_df['quality'].unique())
  for q in range(unique_qualities[0], unique_qualities[-1]):
      ax_subset.scatter(wine_df[wine_df['quality']==q]['fixed acidity'], 
                        wine_df[wine_df['quality']==q]['alcohol'], 
                        wine_df[wine_df['quality']==q]['pH'], label="Quality {}".format(q))
      
  ax_subset.set_xlabel("fixed acidity")
  ax_subset.set_ylabel("alcohol")
  ax_subset.set_zlabel("pH")
  ax_subset.set_box_aspect((np.ptp(wine_df['fixed acidity']), np.ptp(wine_df['alcohol']), np.ptp(wine_df['pH'])))
  plt.title("Wine Subset of Features")
  plt.legend()
  plt.tight_layout()
  plt.show()


wine_database = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv', delimiter=';')

sample_wine = len(wine_database.index)
sample_wines = wine_database.iloc[:, :-1].to_numpy()
qualities = wine_database.iloc[:, -1].to_numpy()
label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(qualities)
labels_wine = label_encoder.transform(qualities)
gmm = {}
gmm['priors'] = (wine_database.groupby(['quality']).size() / sample_wine).to_numpy()
class_length_wine = len(gmm['priors'])
gmm['meanvectors'] = wine_database.groupby(['quality']).mean().to_numpy()
n_wine = gmm['meanvectors'].shape[1]
gmm['covmatrices'] = np.array([covariance_regularization(sample_wines[labels_wine==l].T, (1/n_wine)) for l in range(class_length_wine)])
N_per_l_wine = np.array([sum(labels_wine == l) for l in range(class_length_wine)])
print(N_per_l_wine)

lambda_wine = np.ones((class_length_wine, class_length_wine)) - np.eye(class_length_wine)
decision_map_wine = erm_classification1(sample_wines, lambda_wine, gmm, class_length_wine)
confusion_matrix_wine = confusion1(decision_map_wine, labels_wine)

scatter_plot_wine(wine_database)

def scatter_plot_wine_pca(X, wine_df):
  fig = plt.figure(figsize=(10, 10))

  ax_pca = plt.axes(projection='3d')

  pca = PCA(n_components=3)
  X_fit = pca.fit(X)
  Z = pca.transform(X)
  print("Explained variance ratio: ", pca.explained_variance_ratio_)
  unique_qualities = np.sort(wine_df['quality'].unique())
  for q in range(unique_qualities[0], unique_qualities[-1]):
    ax_pca.scatter(Z[wine_df['quality']==q, 0], 
                      Z[wine_df['quality']==q, 1], 
                      Z[wine_df['quality']==q, 2], label="Quality {}".format(q))
  ax_pca.set_box_aspect((np.ptp(Z[:, 0]), np.ptp(Z[:, 1]), np.ptp(Z[:, 2])))
  ax_pca.set_xlabel(r"$z_1$")
  ax_pca.set_ylabel(r"$z_2$")
  ax_pca.set_zlabel(r"$z_3$")

  plt.title("PCA of Wine Dataset")
  plt.legend()
  plt.tight_layout()
  plt.show()

scatter_plot_wine_pca(sample_wines, wine_database)