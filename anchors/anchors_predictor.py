import numpy as np
import re
from math import inf
import sklearn
from sklearn import datasets
from anchor import anchor_tabular


class AnchorsPredictor:
    """
    Class to perform anchor predictions and data wrangling in order to apply anchors to the data. 
    """
    def __init__(self, anchors):
        """
        Class constructor.
        Parameters
        ----------
        """
        self.anchors = anchors

    def __get_anchor(self, a)-> tuple:
        """
        Function to separate the name of the feature from the ranges.

        Parameters
        ----------
        a : str
            The string containing the anchor.
        Returns
        -------
        anchor : str
            The anchor.
        rest : str
            The rest of the string.
        """
        quoted_part = a.split("'")[1]
        rest = a.replace(f"'{quoted_part}'", '').replace("b", '').strip()

        return quoted_part, rest

    
    def __parse_range(self, expr: str):
        """
        Function to parse the range of the anchor.

        Parameters
        ----------
        expr : str
            The string containing the range.
        Returns
        -------
        low : float
            The lower bound of the range.
        high : float
            The upper bound of the range.
        li : bool
            True if the lower bound is included, False otherwise.
        ui : bool
            True if the upper bound is included, False otherwise.
        """
        expr = expr.strip().replace(" ", "")
        
        patterns = [
            (r"^=(\-?\d+(\.\d+)?)$", 'equals'),
            (r"^(>=|>)\s*(-?\d+(\.\d+)?)$", 'lower'),
            (r"^(<=|<)\s*(-?\d+(\.\d+)?)$", 'upper'),
            (r"^(-?\d+(\.\d+)?)(<=|<){1,2}(<=|<)(-?\d+(\.\d+)?)$", 'between'),
            (r"^(-?\d+(\.\d+)?)(>=|>){1,2}(>=|>)(-?\d+(\.\d+)?)$", 'reverse_between'),
        ]
        
        for pattern, kind in patterns:
            match = re.match(pattern, expr)
            if match:
                if kind == 'equals':
                    num = float(match.group(1))
                    return (num, num, True, True)
                elif kind == 'lower':
                    op, num = match.group(1), float(match.group(2))
                    return (
                        num,
                        inf,
                        op == '>=',
                        False
                    )
                elif kind == 'upper':
                    op, num = match.group(1), float(match.group(2))
                    return (
                        -inf,
                        num,
                        False,
                        op == '<='
                    )
                elif kind == 'between':
                    low = float(match.group(1))
                    op1 = match.group(3)
                    op2 = match.group(4)
                    high = float(match.group(5))
                    return (
                        low,
                        high,
                        op1 == '<=',
                        op2 == '<='
                    )
                elif kind == 'reverse_between':
                    high = float(match.group(1))
                    op1 = match.group(3)
                    op2 = match.group(4)
                    low = float(match.group(5))
                    return (
                        low,
                        high,
                        op2 == '>=',
                        op1 == '>='
                    )

        raise ValueError(f"Unrecognized format: {expr}")

    def __inside(self, val, interval) -> bool:
        """
        Function to check if a value is inside an interval.
        Parameters
        ----------
        val : float
            The value to check.
        interval : tuple
            The interval to check.
        Returns
        -------
        bool
            True if the value is inside the interval, False otherwise.
        """
        low, high, li, ui = interval
        if li and ui:
            return low <= val <= high
        elif li and not ui:
            return low <= val < high
        elif not li and ui:
            return low < val <= high
        else:
            return low < val < high
        
    from typing import Optional, Tuple

    def __intersect(self,
        a: Tuple[float, float, bool, bool],
        b: Tuple[float, float, bool, bool]
    ) -> Optional[Tuple[float, float, bool, bool]]:
        
        a_low, a_high, a_li, a_ui = a
        b_low, b_high, b_li, b_ui = b

        # Compute max of lower bounds
        if a_low > b_low:
            low, li = a_low, a_li
        elif a_low < b_low:
            low, li = b_low, b_li
        else:
            low = a_low
            li = a_li and b_li

        # Compute min of upper bounds
        if a_high < b_high:
            high, ui = a_high, a_ui
        elif a_high > b_high:
            high, ui = b_high, b_ui
        else:
            high = a_high
            ui = a_ui and b_ui

        # Check for empty intersection
        if low > high:
            return None
        if low == high and not (li and ui):
            return None

        return (low, high, li, ui)
    
    def classify(self, input, thresholds, feature_names)-> np.ndarray:
        """
        Classify the input data based on the thresholds.

        Parameters
        ----------
        input : np.ndarray
            The input data to classify.
        thresholds : list of dict
            The thresholds to classify the data.
        feature_names : list of str
            The feature names to classify the data.
        Returns
        -------
        np.ndarray
            The classified data.
        """

        out = np.zeros(input.shape[0])
        
        for i in range(input.shape[0]):
            for j in range(len(thresholds)):
                flag = True
                out[i] = 1
                for nk,k in enumerate(feature_names):
                    if k in thresholds[j]:
                        #print(input[i,nk], thresholds[j][k])
                        if not (self.__inside(input[i,nk], thresholds[j][k])):
                            flag = False
                            out[i] = 0
                            break
                if flag:
                    break
                else:
                    flag = True
            
        return out
    
    def coverage(self, tab, feat_names)-> float:
        """
        Calculate the coverage of the data with respect to the anchors.

        Parameters
        ----------
        tab : list of dict containing the feature intervals for every sample
        feat_names : list of str containing the feature names  for which we want the coverage
        Returns
        -------
        coverage : float
            The coverage of the data with respect to the anchors.
        """
        coverage = 0 
        for i in range(len(tab)):
            product = 1
            for j in feat_names:
                if j in tab[i]:
                    a, b, _, _ = tab[i][j]
                    a = max(a,0)
                    b = min(b,100)
                    length = b - a
                product *= length
            coverage += product
        return coverage/(100**len(feat_names))


    def __grid_points_in_RN(self, explanations, feature_names, delta, STEP, min_vals, max_vals):
        variables = []
        for i in range(len(feature_names)):
            variables.append(np.arange(int(min_vals[i]+STEP), int(max_vals[i]), int(delta)))
        grid_points = np.meshgrid(*variables, indexing='ij')
        grid_points = np.stack([g.ravel() for g in grid_points], axis=-1)
        out = self.classify(grid_points, explanations, feature_names)
        
        idx_del = np.where(out == 1)[0]
        grid_points = np.delete(grid_points, idx_del, axis=0)
        return grid_points
    
    # def __grid_points_in_RN(self, explanations, feature_names, delta, STEP, min_vals, max_vals):
    #     grid_points = [[x1, x2, x3, x4] for x1 in np.arange(0, 100, delta) 
    #                 for x2 in np.arange(0, 100, delta) 
    #                 for x3 in np.arange(0, 100, delta) 
    #                 for x4 in np.arange(0, 100, delta)]
    #     grid_points = np.array(grid_points, dtype=object)
    #     print(grid_points)
    #     out = self.classify(grid_points, explanations, feature_names)
        
    #     idx_del = np.where(out == 0)
    #     np.delete(grid_points, idx_del)
    #     return grid_points

    def __evaluate_grid_points(self, grid_points, models, positive_samples, req_names, min_idx_cf= 0, max_idx_cf = 3):
        merged_points = []
        for grid_point in grid_points:
            for point in positive_samples:
                #print("point:", point)
                #print("grid_point:", grid_point)
                merged_point = np.concatenate((point[min_idx_cf:max_idx_cf+1], grid_point))
                merged_point = np.concatenate((merged_point, [point[-1]]))
                merged_points.append(merged_point)
                #print("merged_point:", merged_point)
        print(len(merged_points))
        
        for r, req in enumerate(req_names):
            print(f"___________Requirement {req}___________")
            
            #classify the samples with the model
            tmp_output = models[r].predict(merged_points)
            if(r == 0):
                output = tmp_output
            else:
                output *= tmp_output

        models_positives = np.where(output != 0)[0]
        merged_points = np.array(merged_points)
        positive_merged_points = merged_points[models_positives]
        print("positive_merged_points:", positive_merged_points.shape)
        return positive_merged_points

    def __create_new_anchors(self, datasets, explainers, models_positives, models, req_number, explanations):
        #print("models_positives:", models_positives[0])
        for i, p_sample in enumerate(models_positives):
            intersected_exp = {}
            print("p_sample:", p_sample)
            for i in range(req_number):
                #get the sample
                #sample = datasets[i].train[p_sample]
                #explain the sample
                exp = explainers[i].explain_instance(np.array(p_sample), models[i].predict, threshold=0.95)
                #get the textual explanation
                exp = exp.names()
                #transform the textual explanations in an interval
                for boundings in exp:
                    quoted, rest = self.__get_anchor(boundings)            
                    if(quoted not in intersected_exp):
                        intersected_exp[quoted] = self.__parse_range(rest)
                    else:
                        intersected_exp[quoted] = self.__intersect(intersected_exp[quoted], self.__parse_range(rest))

        #prepare the data structure
        explanations.append(intersected_exp)
        
        return explanations
    
    def augment_coverage(self, datasets, explanation, feature_names, min_vals, max_vals, models, positive_samples, req_names, explainers, req_number, min_idx_cf= 0, max_idx_cf = 3, delta = 20, STEP = 20, threshold = 0.5):
        coverage = self.coverage(explanation, feature_names)
        while(coverage < threshold):
            grid_points = self.__grid_points_in_RN(explanation, feature_names, delta, STEP, min_vals, max_vals)
            model_positives = self.__evaluate_grid_points(grid_points, models, positive_samples, req_names, min_idx_cf, max_idx_cf)
            explanation = self.__create_new_anchors(datasets, explainers, model_positives, models, req_number, explanation)
            coverage = self.coverage(explanation, feature_names)
            STEP += STEP
        return explanation
    
    def min_dist_polytope(self, x, explanations_table, controllable_features, observable_features): #feature names must be only controllable features if observable features are "satisfied" otherwise
    
        print("x: ", x)
        min_dist_controllable = np.inf
        min_dist_index_controllable = -1

        min_dist_observable = np.inf
        min_dist_index_observable = -1

        contr_f_dist = np.zeros(len(explanations_table))
        obs_f_dist = np.zeros(len(explanations_table))

        print("explanations_table: ", len(explanations_table))
        for i in range(len(explanations_table)):
            print("i: ", i)
            for j, f_name in enumerate(controllable_features):
                a, b = explanations_table[i][f_name][0], explanations_table[i][f_name][1]
                #TODO: METTERE QUESTO NELLE FUNZIONI OFFLINE
                if a == -inf:
                    a = 0
                if b == inf:
                    b = 100
                print("a: ", a)
                print("b: ", b)
                print("x[j]: ", x[j])
                if(x[j] < a):
                    d = (a - x[j]) ** 2
                    contr_f_dist[i] += d
                    print("contr_f_dist[i]: ", contr_f_dist[i])
                    if(contr_f_dist[i] >= min_dist_controllable):
                        print("distanza maggiore della minima controllable")
                        break
                elif(x[j] > b):
                    d = (b - x[j]) ** 2
                    contr_f_dist[i] += d
                    print("contr_f_dist[i]: ", contr_f_dist[i])
                    if(contr_f_dist[i] >= min_dist_controllable):
                        print("distanza maggiore della minima controllable")
                        break
                #elif a <= x[j] <= b:
                #    dist[i] += 0
            if(contr_f_dist[i] < min_dist_controllable):
                min_dist_controllable = contr_f_dist[i]
                min_dist_index_controllable = i
                print("min_dist_controllable: ", min_dist_controllable)
                print("min_dist_index_controllable: ", min_dist_index_controllable)
            print("finito controllable features")
            print("contr_f_dist: ", contr_f_dist)


            for j, f_name in enumerate(observable_features): 
                #NCF start from 3 not from 0, therefor we need to add 3 to the index
                jj = j + 3
                print("f_name", f_name,explanations_table[i])
                a, b = explanations_table[i][f_name][0], explanations_table[i][f_name][1]
                if a == -inf:
                    a = 0
                if b == inf:
                    b = 100
                print("a obs: ", a)
                print("b obs: ", b)
                print("x[j] obs: ", x[jj])

                if(x[jj] < a):
                    d = (a - x[jj]) ** 2
                    #dist[i] += d
                    obs_f_dist[i] += d
                    print("obs_f_dist[i]: ", obs_f_dist[i])
                    if(obs_f_dist[i] >= min_dist_observable):
                        print("distanza maggiore della minima observable")
                        break
                elif(x[jj] > b):
                    d = (b - x[jj]) ** 2
                    #dist[i] += d
                    obs_f_dist[i] += d
                    print("obs_f_dist[i]: ", obs_f_dist[i])
                    if(obs_f_dist[i] >= min_dist_observable):
                        print("distanza maggiore della minima observable")
                        break
                #elif a <= x[j] <= b:
                #    dist[i] += 0
            if(obs_f_dist[i] < min_dist_observable):
                min_dist_observable = obs_f_dist[i]
                min_dist_index_observable = i
                print("min_dist_observable: ", min_dist_observable)
                print("min_dist_index_observable: ", min_dist_index_observable)
            print("finito observable features")
            print("obs_f_dist: ", obs_f_dist)

        return contr_f_dist, obs_f_dist, min_dist_controllable, min_dist_index_controllable, min_dist_observable, min_dist_index_observable


    def evaluate_sample(self, sample, explanation, controllable_features, observable_features):
        print("Sample: ", sample)
        contr_f_dist, obs_f_dist, min_dist_controllable, min_dist_index_controllable, min_dist_observable, min_dist_index_observable = self.min_dist_polytope(sample, explanation, controllable_features, observable_features)
        print("min_dist_controllable: ", min_dist_controllable)
        print("min_dist_index_controllable: ", min_dist_index_controllable)
        print("min_dist_observable: ", min_dist_observable)
        print("min_dist_index_observable: ", min_dist_index_observable)

        if obs_f_dist[min_dist_index_observable] == 0:
            print("The sample is in the polytope for the observable features!")
            if contr_f_dist[min_dist_index_observable] == 0:
                print("The sample is in the polytope for the controllable features too!")
            else:
                print("We are inside polytiope ", min_dist_index_observable, " for the observable features but not for the controllable ones, we will go there")
                polytope = explanation[min_dist_index_observable]
                print("Polytope: ", polytope)   
        else:
            print("The sample is not in the polytope for the observable features!")
            print("The closest polytope for NCF is at dist: ",obs_f_dist[min_dist_index_observable])
        return min_dist_index_controllable, min_dist_index_observable
            