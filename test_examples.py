# -*- coding: utf-8 -*-
__author__ = 'maoss2'
from pyscmGroup import GroupSCM
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
import argparse
import logging
import random
import numpy as np 
random.seed(42)
logging.getLogger('parso.python.diff').disabled = True
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nb_jobs = 2
cv_fold = KFold(n_splits=3, random_state=42)

param_model_type = ['conjunction', 'disjunction']
param_p = [0.1, 0.316, 0.45, 0.562, 0.65, 0.85, 1.0, 2.5, 4.39, 5.623, 7.623, 10.0]
param_max_attributes = np.arange(1, 7, 1)
parameters_group_scm = {'model_type': param_model_type,
                        'p': param_p,
                        'max_rules': param_max_attributes
                        }
def get_metrics(y_test, predictions_binary):
    """Compute the metrics for classifiers predictors
    Args:
        y_test: real labels
        predictions_binary: the predicted labels
    Return: metrics: a dictionnary of the metrics
    """
    y_test = np.asarray(y_test, dtype=np.float)
    predictions_binary = np.asarray(predictions_binary, dtype=np.float)
    metrics = {"accuracy": accuracy_score(y_test, predictions_binary),
               "f1_score": f1_score(y_test, predictions_binary),
               "precision": precision_score(y_test, predictions_binary),
               "recall": recall_score(y_test, predictions_binary)
               }
    return metrics


def f_1(c, x):
    """Compute an update function"""
    return np.exp( -c * x) 

def build_priors_rules_vector(c,
                              activation_function=f_1,
                              random_weights = False,
                              dictionnary_for_prior_group={}, 
                              dictionnary_for_prior_rules={}):
    """
    Build the vector of the prior rules integreting the prior on the group/pathways 
    Args:
        c, an hp
        dictionnary_for_prior_group, str, path to the dictionnary for generating . Structure must be: d = {'Group_name1': [gen1, gen100,...],  'Group_nameX': [genXXX,...]}
        dictionnary_for_prior_rules, str, path to the dictionnary. Structure must be: d = {'Feature_name1': [Group_name1, Group_name100,...],  'Feature_nameX': [Group_nameXXX,...]}
    Return:
        prior_values_dict_pr_group, dict
        prior_values_dict_pr_rules, dict
    """   
    dict_pr_group = dictionnary_for_prior_group
    dict_pr_rules = dictionnary_for_prior_rules
    # Build PriorGroups vector, p_g
    prior_values_dict_pr_group = {k: f_1(c, len(v)) for k, v in dict_pr_group.items()} 
    # Build PriorRules vector, p_ri
    if random_weights:
        random.seed(42)
        np.random.seed(42)
        values_randomly_generated = np.random.rand(len(dict_pr_group.items()))
        prior_values_dict_pr_group = {k: activation_function(c, values_randomly_generated[idx]) for idx, k in enumerate(dict_pr_group.keys())}
        prior_values_dict_pr_rules = {k: activation_function(c, np.sum([prior_values_dict_pr_group[el] for el in v])) for k, v in dict_pr_rules.items()}
    else:
        prior_values_dict_pr_rules = {k: activation_function(c, np.sum([prior_values_dict_pr_group[el] for el in v])) for k, v in dict_pr_rules.items()}
    return prior_values_dict_pr_group, prior_values_dict_pr_rules

n_samples = 2000
n_features = 25
X, y = make_classification(n_samples=n_samples,
                           n_features=n_features,
                           n_informative=3,
                           n_redundant=0,
                           n_repeated=0,
                           n_classes=2,
                           random_state=0,
                           shuffle=False)
features_names = [f'feature_{i}' for i in range(n_features)]
features_to_index = {idx: name for idx, name in enumerate(features_names)}
group_feat_1 = random.choices(features_names, k=5)
group_feat_2 = random.choices(features_names, k=5)
temp_fusion = deepcopy(group_feat_1)
temp_fusion.extend(group_feat_2)
group_feat_3 = [el for el in features_names if el not in temp_fusion]
dictionnary_for_prior_group = {'G_1':group_feat_1, 'G_2':group_feat_2, 'G_3':group_feat_3,}
list_of_groups = ['G_1', 'G_2', 'G_3']
dictionnary_for_prior_rules = {feat_name: random.sample(list_of_groups, k=random.choice([1, 2, 3])) for feat_name in features_names}
_, prior_values_dict_pr_rules = build_priors_rules_vector(c=0.1,
                                                          activation_function=f_1,
                                                          random_weights=False, 
                                                          dictionnary_for_prior_group=dictionnary_for_prior_group, 
                                                          dictionnary_for_prior_rules=dictionnary_for_prior_rules)
prior_rules = [prior_values_dict_pr_rules[name] for name in features_names]

learner = GroupSCM(features_to_index=features_to_index, 
                        prior_rules=prior_rules, 
                        update_method='inner_group',
                        groups=dictionnary_for_prior_rules,
                        tiebreaker='', 
                        p=1.0, 
                        model_type='conjunction', 
                        max_rules=3)
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

gs_clf = GridSearchCV(learner, param_grid=parameters_group_scm, n_jobs=nb_jobs, cv=cv_fold, verbose=1)

gs_clf.fit(x_train, y_train)
pred = gs_clf.predict(x_test)
y_train_pred = gs_clf.predict(x_train)
train_metrics = get_metrics(y_test=y_train, predictions_binary=y_train_pred)
print(learner)
print('*' * 50)
print('Train metrics', train_metrics)
metrics = get_metrics(y_test=y_test, predictions_binary=pred)
print('Test metrics', metrics)
print()
cnf_matrix = confusion_matrix(y_test, pred)
print(cnf_matrix)
rules_retrieved = gs_clf.best_estimator_.get_stats()
print(rules_retrieved)
for el in rules_retrieved['Binary_attributes']:
    print(f'{(el.__str__(), features_names[el.feature_idx], dictionnary_for_prior_rules[features_names[el.feature_idx]])}')