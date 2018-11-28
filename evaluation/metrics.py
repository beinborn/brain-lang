from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn import preprocessing
import numpy as np
import itertools

import scipy.spatial as sp
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import logging




def mse(predictions, targets):
  """Mean Squared Error.
  :param predictions: (n_samples, n_outputs)
  :param targets: (n_samples, n_outputs)
  :return:
    a scalar which is mean squared error
  """
  return  mean_squared_error(predictions, targets)

def pairwise_matches(prediction1, target1, prediction2, target2):
    matches = {}
    for similarity_metric_fn in [cosine_similarity, euclidean_similarity, pearson_correlation]:
        correct1 = similarity_metric_fn(prediction1, target1)
        false1 = similarity_metric_fn(prediction1, target2)
        correct2 = similarity_metric_fn(prediction2, target2)
        false2 = similarity_metric_fn(prediction2, target1)

        metric_name = similarity_metric_fn.__name__
        matches[metric_name +"_Mitchell"] = int((correct1 + correct2) > (false1 + false2))
        matches[metric_name + "_Wehbe1"] = int(correct1 > false1)
        matches[metric_name + "_Wehbe2"] = int(correct2 > false2)
        matches[metric_name + "_Strict"] = int((correct1 >false1) & (correct2 > false2))
    return matches

# def pairwise_cosine_match_mitchell(prediction1, target1, prediction2, target2):
#     return (cosine_similarity(prediction1, target1) + cosine_similarity(prediction2, target2)) > (cosine_similarity(prediction1, target2) + cosine_similarity(prediction2, target1))
#
#
# def pairwise_cosine_match_me(prediction1, target1, prediction2, target2):
#
#     match1 = cosine_similarity(prediction1, target1) > cosine_similarity(prediction1, target2)
#     match2 = cosine_similarity(prediction2, target2) > cosine_similarity(prediction2, target1)
#
#     return match1 & match2
#
# def pairwise_cosine_match_wehbe(prediction1, target1, prediction2, target2):
#
#     match1 =  cosine_similarity(prediction1, target1) > cosine_similarity(prediction1, target2)
#     match2 = cosine_similarity(prediction2, target2) > cosine_similarity(prediction2, target1)
#
#     return match1 , match2

def first_order_rdm(data, distance_metric):
    RDMs = []
    for i in range(len(data)):
      RDM = sp.distance.cdist(np.asarray(condition_data[i]),np.asarray(condition_data[i]), distance_metric)
      RDMs.append(RDM)
    return RDMs

def second_order_rdm(RDMs):
    flat = [m.flatten(1) for m in RDMs]
    flat = np.array(flat).transpose()
    c_matrix = spearmanr(flat)[0]
    if not(isinstance(c_matrix, np.ndarray)):
        c_matrix = np.array([[1,c_matrix],[c_matrix,1]])
    RDM = np.ones(c_matrix.shape) - c_matrix
    return RDM

def delete_constant_rows(predictions, targets):
    index = 0
    num_deleted = 0
    for i in targets:
        if (np.var(i) == 0):
            targets = np.delete(targets, index, 0)
            predictions = np.delete(predictions, index, 0)
            num_deleted += 1
        index += 1
    logging.info("Ignoring " + str(num_deleted) + " constant voxels in result.")
    return predictions, targets


def plot_RDM(RDMs):

        fig, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(RDMs, cmap='Spectral', xticklabels=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 'embeds'],
                         yticklabels=['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 'embeds'], ax=ax)
        ax.set_title("Correlated distances between scans and embedding matrices")
        # fig.savefig("plots/RDM_2nd_order.png", bbox_inches='tight',dpi=250)
### EVALUATION METHODS ###

def cosine_similarity(vector1, vector2):

    return 1 - sp.distance.cosine(vector1, vector2)

def euclidean_similarity(vector1, vector2):
    return 1 - sp.distance.euclidean(vector1, vector2)

def pearson_correlation(vector1,vector2):
    return pearsonr(vector1, vector2)[0]



def explained_variance(predictions, targets, multioutput="raw_values"):
    return explained_variance_score(predictions, targets, multioutput=multioutput)


def sum_explained_variance(predictions, targets):
    ev_scores = explained_variance(predictions, targets, multioutput="raw_values")
    return np.sum(np.asarray(ev_scores))

def mean_explained_variance(predictions, targets):
    return explained_variance(predictions, targets, multioutput="uniform_average")

def sum_r2(predictions, targets):
     r2values = r2_score(predictions, targets,  multioutput="raw_values")
     return np.sum(np.asarray(r2values))


def mean_r2(predictions, targets):
    return r2_score(predictions, targets, multioutput="uniform_average")

def mean_r2_for_topn(predictions, targets, n = 50):
    r2values = r2_score(predictions, targets, multioutput="raw_values")
    top_n = sorted(r2values, reverse = True)[0:n]
    return np.mean(np.asarray(top_n))
def mean_ev_for_topn(predictions, targets, n = 50):
    evvalues = explained_variance(predictions, targets, multioutput="raw_values")
    top_n = sorted(evvalues, reverse = True)[0:n]
    return np.mean(np.asarray(top_n))

# TODO not yet working, debug
def representational_similarity_analysis(scans, embeddings, distance_metric='cosine'):
    # If we do this, we do not need to apply a mapping model
    # Calculate matrix of distances between all pairs of scans
    # Calculate matrix of distances between all pairs of embeddings
    l = scans + embeddings
    RDMS = first_order_rdm(l, distance_metric)
    # for all possible pairs of (scan, embedding) calculate
    # correlation between distance vectors in similarity metrics
    RDM = second_order_rdm(RDMS)




### HAVE NOT LOOKED AT THIS YET ####
def search_light_analysis(predictions, targets, voxel_to_xyz_mapping):
    raise NotImplementedError()




