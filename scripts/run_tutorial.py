#%%

"""
Welcome to the S2AND tutorial!

We will cover a few aspects of the S2AND pipeline:

(1) Load a test dataset.
(2) Fit a pairwise model + bells & whistles.
(3) Fit a clusterer.
(4) Evaluate the pairwise model and the clusterer.
"""

#%%

import os

os.environ["OMP_NUM_THREADS"] = "8"

import json
import copy
import argparse
import logging
import pickle
from typing import Dict, Any, Optional, List
from collections import defaultdict

import numpy as np
import pandas as pd

from s2and.data import ANDData
from s2and.featurizer import featurize, FeaturizationInfo
from s2and.model import PairwiseModeler, Clusterer, FastCluster
from s2and.eval import pairwise_eval, cluster_eval, facet_eval
from s2and.consts import FEATURIZER_VERSION, DEFAULT_CHUNK_SIZE, PROJECT_ROOT_PATH
from s2and.file_cache import cached_path
from s2and.plotting_utils import plot_facets
from hyperopt import hp



# this is the random seed we used for the ablations table
random_seed = 42
# number of cpus to use
n_jobs = 4


#%%

# we're going to load the arnetminer dataset
# and assume that you have it already downloaded to the `S2AND/data/` directory

dataset_name = 'arnetminer'
DATA_DIR = os.path.join(PROJECT_ROOT_PATH, 'data', dataset_name)

anddata = ANDData(
    signatures=os.path.join(DATA_DIR, dataset_name + "_signatures.json"),
    papers=os.path.join(DATA_DIR, dataset_name + "_papers.json"),
    name=dataset_name,
    mode="train",  # can also be 'inference' if just predicting
    specter_embeddings=os.path.join(DATA_DIR, dataset_name + "_specter.pickle"),
    clusters=os.path.join(DATA_DIR, dataset_name + "_clusters.json"),
    block_type="s2",  # can also be 'original'
    train_pairs=None,  # in case you have predefined splits for the pairwise models
    val_pairs=None,
    test_pairs=None,
    train_pairs_size=100000,  # how many training pairs for the pairwise models?
    val_pairs_size=10000,
    test_pairs_size=10000,
    n_jobs=n_jobs,
    load_name_counts=True,  # the name counts derived from the entire S2 corpus need to be loaded separately
    preprocess=True,
    random_seed=random_seed,
)



#%%

# to train the pairwise model, we define which feature categories to use
# here it is all of them
features_to_use = [
    "name_similarity",
    "affiliation_similarity",
    "email_similarity",
    "coauthor_similarity",
    "venue_similarity",
    "year_diff",
    "title_similarity",
    "reference_features",
    "misc_features",
    "name_counts",
    "embedding_similarity",
    "journal_similarity",
    "advanced_name_similarity",
]

# we also have this special second "nameless" model that doesn't use any name-based features
# it helps to improve clustering performance by preventing model overreliance on names
nameless_features_to_use = [
    feature_name
    for feature_name in features_to_use
    if feature_name not in {"name_similarity", "advanced_name_similarity", "name_counts"}
]

# we store all the information about the features in this convenient wrapper
featurization_info = FeaturizationInfo(features_to_use=features_to_use, featurizer_version=FEATURIZER_VERSION)
nameless_featurization_info = FeaturizationInfo(features_to_use=nameless_features_to_use, featurizer_version=FEATURIZER_VERSION)

# now we can actually go and get the pairwise training, val and test data
train, val, test = featurize(anddata, featurization_info, n_jobs=4, use_cache=False, chunk_size=DEFAULT_CHUNK_SIZE, nameless_featurizer_info=nameless_featurization_info, nan_value=np.nan)  # type: ignore
X_train, y_train, nameless_X_train = train
X_val, y_val, nameless_X_val = val
X_test, y_test, nameless_X_test = test



#%%

# now we define and fit the pairwise modelers
pairwise_modeler = PairwiseModeler(
    n_iter=25,  # number of hyperparameter search iterations
    estimator=None,  # this will use the default LightGBM classifier
    search_space=None,  # this will use the default LightGBM search space
    monotone_constraints=featurization_info.lightgbm_monotone_constraints,  # we use monotonicity constraints to make the model more sensible
    random_state=random_seed,
)
pairwise_modeler.fit(X_train, y_train, X_val, y_val)


# as mentioned above, there are 2: one with all features and a nameless one
nameless_pairwise_modeler = PairwiseModeler(
    n_iter=25,
    estimator=None,
    search_space=None,
    monotone_constraints=nameless_featurization_info.lightgbm_monotone_constraints,
    random_state=random_seed,
)
nameless_pairwise_modeler.fit(nameless_X_train, y_train, nameless_X_val, y_val)



#%%

# now we can fit the clusterer itself
clusterer = Clusterer(
    featurization_info,
    pairwise_modeler.classifier,  # the actual pairwise classifier
    cluster_model=FastCluster(linkage='average'),  # average linkage agglomerative clustering
    search_space={"eps": hp.uniform("choice", 0, 1)},  # the hyperparemetrs for the clustering algorithm
    n_jobs=n_jobs,
    use_cache=False,
    nameless_classifier=nameless_pairwise_modeler.classifier,  # the nameless pairwise classifier
    nameless_featurizer_info=nameless_featurization_info,
    random_state=random_seed,
    use_default_constraints_as_supervision=False,  # this is an option used by the S2 production system but not in the S2AND paper
)
clusterer.fit(anddata)


#%%

# but how good are our models?
# first, let's look at the quality of the pairwise evaluation
pairwise_metrics = pairwise_eval(
    X_test,
    y_test,
    pairwise_modeler,
    os.path.join(PROJECT_ROOT_PATH, "data", "tutorial_figures"),  # where to put the figures
    "tutorial_figures",  # what to call the figures
    featurization_info.get_feature_names(),
    nameless_classifier=nameless_pairwise_modeler,
    nameless_X=nameless_X_test,
    nameless_feature_names=nameless_featurization_info.get_feature_names(),
    skip_shap=False,  # if your model isn't a tree-based model, you should put True here and it will not make SHAP figures
)
print(pairwise_metrics)

#%%

# we can do the same thing for the clustering performance
cluster_metrics, b3_metrics_per_signature = cluster_eval(
    anddata,
    clusterer,
    split="test",  # which part of the data to evaluate on, can also be 'val'
    use_s2_clusters=False,  # set to true if you want to see how the old S2 system does
)
print(cluster_metrics)

# if you want to reproduce the facet plots from the S2AND paper, check out the `facet_eval` function also!
