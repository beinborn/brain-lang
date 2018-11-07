from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
import numpy as np
import itertools

def mean_squared_error(predictions, targets):
  """

  :param predictions: (n_samples, n_outputs)
  :param targets: (n_samples, n_outputs)
  :return:
    a scalar which is mean squared error
  """
  return  mean_squared_error(predictions, targets)

def explained_variance(predictions, targets):
  return explained_variance_score(targets, predictions, multioutput='raw_values')

def mean_explain_variance(predictions, targets):
  return explained_variance_score(targets, predictions, multioutput='uniform_average')

def cosine_similarity(predictions, targets):


def binary_accuracy(predictions, targets):

def binary_accuracy_from_dists(cosine_dists, euclidean_dists):
  nn_index = np.argmin(cosine_dists, axis=1)
  accuracy_on_test = np.mean(nn_index == np.argmax(np.eye(cosine_dists.shape[0]), axis=1))

  b_acc = []
  e_b_acc = []
  for i, j in itertools.combinations(np.arange(cosine_dists.shape[0]), 2):
    right_match = cosine_dists[i, i] + cosine_dists[j, j]
    wrong_match = cosine_dists[i, j] + cosine_dists[j, i]
    b_acc.append(right_match < wrong_match)

    e_right_match = euclidean_dists[i, i] + euclidean_dists[j, j]
    e_wrong_match = euclidean_dists[i, j] + euclidean_dists[j, i]
    e_b_acc.append(e_right_match < e_wrong_match)

  return np.mean(b_acc), np.mean(e_b_acc), b_acc, e_b_acc

def MRR(distances):
  prec_at_corrects = []
  ranks = []
  sorted_indexes = np.argsort(distances, axis=1)
  for i in np.arange(len(distances)):
    # print(i)
    correct_at = np.where(sorted_indexes[i] == i)[0] + 1
    # print("Reciprocal Rank",correct_at)
    prec_at_correct = 1.0 / correct_at
    # print("precision at ",correct_at,": ",prec_at_correct)
    prec_at_corrects.append(prec_at_correct)
    ranks.append(correct_at)

  print("MRR: ", np.mean(prec_at_corrects), " ", np.mean(ranks))
  return np.mean(ranks), np.mean(prec_at_corrects), ranks, prec_at_corrects

