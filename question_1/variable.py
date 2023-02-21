import numpy as np

m0 = np.array([-0.5, -0.5, -0.5, -0.5]).reshape(-1, 1)
C0 = np.array([[2/4, -0.5/4, 0.3/4, 0], [-0.5/4, 1/4, -0.5/4, 0], [0.3/4, -0.5/4, 1/4, 0], [0, 0, 0, 2/4]])
m1 = np.array([1, 1, 1, 1]).reshape(-1, 1)
C1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])
priors = np.array([0.35, 0.65])
gmmParameters_q1 = {'priors': priors, 'meanVectors': np.hstack((m0, m1)), 'covMatrices': np.dstack((C0, C1))}
N = 10000

sample_path = 'C:\\Users\\omkar\\OneDrive\\Desktop\\eece5644\\HW1\\question_1\\sample_list.csv'
class_length_q1 = len(priors)
L_q1 = np.array(range(class_length_q1))

meanvectors_q1 = np.array([[-0.5, -0.5, -0.5, -0.5], [1, 1, 1, 1]])
covmatrices_q1 = np.array([[[2/4, -0.5/4, 0.3/4, 0], [-0.5/4, 1/4, -0.5/4, 0], [0.3/4, -0.5/4, 1/4, 0], [0, 0, 0, 2/4]],[[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]]])