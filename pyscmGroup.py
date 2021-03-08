# -*- coding: utf-8 -*-
from __future__ import division

import logging

import numpy as np
from future.utils import iteritems
from pyscm._scm_utility import find_max as find_max_utility  # cpp extensions
from pyscm.model import ConjunctionModel, DisjunctionModel
from pyscm.rules import DecisionStump
from pyscm.scm import BaseSetCoveringMachine
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_random_state, check_X_y

__author__ = 'maoss2'


class GroupSetCoveringMachineClassifier(BaseSetCoveringMachine):
    """This version works without features duplication """
    def __init__(self, features_to_index, prior_rules, update_method, groups, tiebreaker, p=1.0, 
                 model_type="conjunction", max_rules=10, random_state=None):
        super(GroupSetCoveringMachineClassifier, self).__init__(p=p, model_type=model_type,
                                                                max_rules=max_rules, random_state=random_state)
        """ 
        Args:
            features_to_index: a dictionnary with key= idx and value the correspondant features at place idx 
            prior_rules : The prior or preference on the rules (pre-calculated)
            update_method: str, name of the method to update the prioRules in the GroupSCM model
                'inner_group': p_ri = p_ri * exp(| g_i \intersection GR | )  update 1
                'outer_group': p_ri = p_ri * exp(-| g_i \intersection GR | )  update 2
                
            groups : g_i \in [1, G]+ is the set of groups associated with the rule r_i, 
                where G is the total number of groups. 
                More explicitly each rule can have multiple groups/ pathways
        
        """
        self.features_to_index = features_to_index
        self.prior_rules = np.asarray(prior_rules)
        self.groups = groups
        self.tiebreaker = tiebreaker
        self.update_method = update_method
        self.groups_rules = [] # GR
        
    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        """
        Fit a SCM model.

        Args:
            X: array-like, shape=[n_examples, n_features]
                The feature of the input examples.
            y : array-like, shape = [n_samples]
                The labels of the input examples.
            tiebreaker: a function defines by the user, If None, the rule that most
                decreases the training error is selected.
            iteration_callback: function(model)
                A function that is called each time a rule is added to the model.

        Returns:
            self: object
                Returns self.
        """
        # random_state = check_random_state(self.random_state)

        if self.model_type == "conjunction":
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif self.model_type == "disjunction":
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")

        # Initialize callbacks
        if iteration_callback is None:
            iteration_callback = lambda x: None

        # Parse additional fit parameters
        logging.debug("Parsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in iteritems(fit_params):
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        # Validate the input data
        logging.debug("Validating the input data")
        X, y = check_X_y(X, y)
        X = np.asarray(X, dtype=np.double)
        self.classes_, y, total_n_ex_by_class = np.unique(y, return_inverse=True, return_counts=True)
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        logging.debug("The data contains %d examples. Negative class is %s (n: %d) and positive class is %s (n: %d)." %
                      (len(y), self.classes_[0], total_n_ex_by_class[0], self.classes_[1], total_n_ex_by_class[1]))

        # Invert the classes if we are learning a disjunction
        logging.debug("Preprocessing example labels")
        pos_ex_idx, neg_ex_idx = self._get_example_idx_by_class(y)
        y = np.zeros(len(y), dtype=np.int)
        y[pos_ex_idx] = 1
        y[neg_ex_idx] = 0

        # Presort all the features
        logging.debug("Presorting all features")
        X_argsort_by_feature_T = np.argsort(X, axis=0).T.copy()

        # Create an empty model
        logging.debug("Initializing empty model")
        self.model_ = ConjunctionModel() if self.model_type == "conjunction" else DisjunctionModel()

        logging.debug("Training start")
        remaining_example_idx = np.arange(len(y))
        remaining_negative_example_idx = neg_ex_idx
        
        features_weights = self.prior_rules # Initialization of groups weights 
        # Feature_weights is the vector used to multipled the utility function in find_max_utility (update_optimal_solution)
        while len(remaining_negative_example_idx) > 0 and len(self.model_) < self.max_rules:
            iteration_info = {"iteration_number": len(self.model_) + 1}
            iteration_info['list_feat_idx_selected'] = []
            logging.debug("Finding the optimal rule to add to the model")
            opti_utility, \
            opti_feat_idx, \
            opti_threshold, \
            opti_kind, \
            opti_N, \
            opti_P_bar = self._get_best_utility_rules(X,
                                                      y,
                                                      X_argsort_by_feature_T,
                                                      remaining_example_idx.copy(),
                                                      groups=self.groups,
                                                      groups_rules=self.groups_rules,
                                                      features_weights=features_weights,
                                                      next_rule_model_idx=self.model_.rules[-1].feature_idx if len(self.model_.rules) != 0 else None,  # int or NoneType
                                                      **utility_function_additional_args)
            iteration_info["utility_max"] = opti_utility
            iteration_info["utility_argmax"] = opti_feat_idx
            logging.debug("Greatest utility is %.5f" % iteration_info["utility_max"])
            logging.debug("There are %d rules with the same utility." % len(iteration_info["utility_argmax"]))
            # Do not select rules that cover no negative examples and make errors on no positive examples
            best_utility_idx = iteration_info["utility_argmax"][
                np.logical_or(opti_N != 0, opti_P_bar != 0)]
            iteration_info['best_utility_idx'] = best_utility_idx
            logging.debug("Tiebreaking. Found %d optimal rules" % len(opti_feat_idx))
            # Let's use the default tiebreaker first
            if len(opti_feat_idx) > 1:
                if tiebreaker is None:
                    training_risk_decrease = 1.0 * opti_N - opti_P_bar
                    keep_idx = np.where(training_risk_decrease == training_risk_decrease.max())[0][0]
                else:  # Tiebreaking using the argmin
                    if len(self.model_.rules) == 0:
                        keep_idx = 0
                    else:
                        list_model_idx = [el.feature_idx for el in self.model_.rules]
                        rules_to_untie = [self.features_to_index[idx] for idx in list_model_idx]
                        print(f'Yeah we use the tiebreaking and the rules to untie are {rules_to_untie}')
                        length_groups_of_rules_to_untie = [len(self.groups[rule]) for rule in rules_to_untie]
                        keep_idx = np.argmin(length_groups_of_rules_to_untie)
            else:
                keep_idx = 0    
            stump = DecisionStump(feature_idx=opti_feat_idx[keep_idx], threshold=opti_threshold[keep_idx],
                                  kind="greater" if opti_kind[keep_idx] == 0 else "less_equal")

            logging.debug("The best rule has utility %.3f" % opti_utility)
            self._add_attribute_to_model(stump)

            logging.debug("Discarding all examples that the rule classifies as negative")
            remaining_example_idx = remaining_example_idx[stump.classify(X[remaining_example_idx])]
            remaining_negative_example_idx = remaining_negative_example_idx[stump.classify(X[remaining_negative_example_idx])]
            logging.debug("There are %d examples remaining (%d negatives)" % (len(remaining_example_idx),
                                                                              len(remaining_negative_example_idx)))

            iteration_callback(self.model_)

        logging.debug("Training completed")

        self.rule_importances_ = []  # TODO: implement rule importances (like its done in Kover)

        return self
    
    def _get_best_utility_rules(self, X, y, X_argsort_by_feature_T, example_idx, groups, groups_rules, features_weights,  next_rule_model_idx):
        if next_rule_model_idx is None:
            # calculated_utility = find_max_utility(self.p, X, y, X_argsort_by_feature_T, example_idx, features_weights)
            # print(f'the initial utility function is: {calculated_utility}')
            return find_max_utility(self.p, X, y, X_argsort_by_feature_T, example_idx, features_weights)
        else:
            previous_rule_choosed_i = self.features_to_index[next_rule_model_idx] # must give a feature Name
            # UPDATE PRI FOR ALL THE RULES HAVING AT LEAST ONE OF THE GROUPS OF THE PREVIOUS RULES
            groups_previous_rule_choosed_i = self.groups[previous_rule_choosed_i] # list 
            # print(f'the groups of the choosen rules are: {groups_previous_rule_choosed_i}')
            # retrieve the intersection for each rules candidates/features: dict of keys: features_name, values: int
            dict_intersection_groups_rules = {feat_name: len([el for el in groups_previous_rule_choosed_i if el in groups_list]) for feat_name, groups_list in self.groups.items()}
            if self.update_method == 'inner_group':
                for idx, feat_name_idx in self.features_to_index.items():
                    if idx != next_rule_model_idx: # pas la peine de reponderer la meme regle quoi
                        features_weights[idx] *= np.exp(dict_intersection_groups_rules[feat_name_idx])
            elif self.update_method == 'outer_group':
                for idx, feat_name_idx in self.features_to_index.items():
                    if idx != next_rule_model_idx: # pas la peine de reponderer la meme regle quoi
                        features_weights[idx] *= np.exp(- dict_intersection_groups_rules[feat_name_idx])    
            else:
                 raise ValueError(f"{self.update_method} must be a str and in ['inner_group', 'outer_group']")
            # UPDATE GR
            # self.groups_rules.extend(groups_previous_rule_choosed_i) # Expected results: [G1, G2, ...]
            # self.groups_rules = list(np.unique(self.groups_rules))
            # calculated_utility = find_max_utility(self.p, X, y, X_argsort_by_feature_T, example_idx, features_weights)
            # print(f'the new utility function is: {calculated_utility}')
            return find_max_utility(self.p, X, y, X_argsort_by_feature_T, example_idx, features_weights)
        
        

class GroupSCM(BaseEstimator, ClassifierMixin):
    """
    A hands on class of GroupSetCoveringMachineClassifier, built with sklearn format in order to use sklearn function on SCM like
    CV, gridsearch, and so on ...
    """

    def __init__(self, features_to_index, prior_rules, update_method, groups, tiebreaker='', 
                 model_type='conjunction', p=0.1, max_rules=10, random_state=42):
        super(GroupSCM, self).__init__()
        self.model_type = model_type
        self.p = p
        self.max_rules = max_rules
        self.random_state = random_state
        self.features_to_index = features_to_index
        self.prior_rules = prior_rules
        self.groups = groups
        self.tiebreaker = tiebreaker
        self.update_method = update_method
        
    def fit(self, X, y):
        self.clf = GroupSetCoveringMachineClassifier(features_to_index=self.features_to_index, 
                                                    prior_rules=self.prior_rules,
                                                    update_method=self.update_method, 
                                                    groups=self.groups, 
                                                    tiebreaker=self.tiebreaker,
                                                    model_type=self.model_type, 
                                                    max_rules=self.max_rules, 
                                                    p=self.p, 
                                                    random_state=self.random_state)
        self.clf.fit(X=X, y=y)
        self.groups_rules = self.clf.groups_rules

    def predict(self, X):
        return self.clf.predict(X)

    def set_params(self, **params):
        for key, value in iteritems(params):
            if key == 'p':
                self.p = value
            if key == 'model_type':
                self.model_type = value
            if key == 'max_rules':
                self.max_rules = value
            if key == 'features_to_index':
                self.features_to_index = value
            if key == 'prior_rules':
                self.prior_rules = value
            if key == 'update_method':
                self.update_method = value
            if key == 'groups':
                self.groups = value
            if key == 'tiebreaker':
                self.tiebreaker = value
        return self

    def get_stats(self):
        return {"Binary_attributes": self.clf.model_.rules}
