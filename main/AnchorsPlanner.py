import numpy as np
import re
from math import inf
import sklearn
from sklearn import datasets
from sklearn.metrics import accuracy_score

from anchor import utils
from anchor import anchor_tabular
import pandas as pd
from multiprocessing import Pool, cpu_count

from util import vecPredictProba



class AnchorsPlanner:
    """
    Class to perform anchor predictions and data wrangling in order to apply anchors to the data. 
    """
    def __init__(self, training_dataset, reqClassifiers, reqNames, 
                 anchorsConfidence, feature_number, feature_names,
                 controllableFeatureIndices, controllableFeatureDomains ):
        """
        Class constructor.
        Parameters
        ----------
        """
        self.reqClassifiers = reqClassifiers
        self.reqNames = reqNames
        self.feature_number = feature_number
        self.feature_names = feature_names
        self.anchorsConfidence = anchorsConfidence
        self.controllableFeatureIndices = np.array(controllableFeatureIndices)
        self.controllableFeaturesNames = [feature_names[i] for i in controllableFeatureIndices]

        self.observableFeatureIndices = [i for i in range(feature_number) if i not in controllableFeatureIndices]
        self.observableFeaturesNames = [feature_names[i] for i in self.observableFeatureIndices]
        self.controllableFeatureDomains = controllableFeatureDomains

        print("Requirements classifiers: ", reqClassifiers)

        training_df = pd.read_csv(training_dataset)
        print("Training dataset loaded: ", training_df.shape)
        all_true_training = training_df[
            (training_df['req_0'] == 1) &
            (training_df['req_1'] == 1) &
            (training_df['req_2'] == 1) &
            (training_df['req_3'] == 1)
        ].drop(columns=reqNames)
        print("Training dataset with all requirements satisfied: ", all_true_training.shape)

        datasets = []
        true_from_anchors_df = {}
        features_to_use = [i for i in range(feature_number)]

        for i,r in enumerate(reqNames):
            datasets.append(\
                utils.load_csv_dataset(\
                    training_dataset, feature_number+i,\
                    features_to_use=features_to_use,\
                    categorical_features=None))
            
            true_from_anchors_df[r] = np.nonzero(datasets[i].labels_train)[0]
            print('Training samples with {} satisfied: '.format(r), true_from_anchors_df[r].shape, "over ", datasets[i].train.shape[0])

        explainer = []
        req_number = len(reqNames)
        for i in range(req_number):
            #initialize the explainer
            explainer.append(anchor_tabular.AnchorTabularExplainer(
                datasets[i].class_names, #it maps the 0 and 1 in the dataset's requirements to the class names
                datasets[i].feature_names,
                datasets[i].train,
                datasets[i].categorical_names)) 
        print("Datasets[0]: ", datasets[0].train.shape, datasets[0].labels_train.shape)
        print("Datasets[1]: ", datasets[1].train.shape, datasets[1].labels_train.shape)
        print("Datasets[2]: ", datasets[2].train.shape, datasets[2].labels_train.shape)
        print("Datasets[3]: ", datasets[3].train.shape, datasets[3].labels_train.shape)

        for i in range(req_number):
            print(f"Model {i+1} training accuracy: {accuracy_score(datasets[i].labels_train, reqClassifiers[i].predict(datasets[i].train)):.4f}")

        for i, req in enumerate(reqNames):
            print(f"___________Requirement {i+1}: {req}___________")
            output = reqClassifiers[i].predict(datasets[i].train)
            #obtain the indices of the samples that have the requirement satisfied (truly in the dataset)
            real_values_single_req = np.where(datasets[i].labels_train == 1)[0]

            if(i == 0):
                final = np.where(output == 1)[0]
                real_values = real_values_single_req
            else:
                final = np.intersect1d(final, np.where(output == 1)[0]) 
                real_values = np.intersect1d(real_values, real_values_single_req)


        positively_classified = final
        print(f"Number of samples with all requirements satisfied (according to model): {positively_classified.shape[0]}")

        print(f"Number of samples with all requirements satisfied (real data): {real_values.shape[0]}")
        #calculate false positives
        f_p = positively_classified.shape[0]- np.intersect1d(real_values, positively_classified).shape[0]
        print(f"Number of false positives from model: {f_p}")
        #calculate the missclassified real positive
        m_r_p = real_values.shape[0] - np.intersect1d(real_values, positively_classified).shape[0]
        print(f"Number of missclassified real positives: {m_r_p}")

        final = np.ones(datasets[0].train.shape[0])
        real_values = np.ones(datasets[0].train.shape[0])
        for i, req in enumerate(reqNames):
            print(f"___________Requirement {i+1}: {req}___________")
            output = reqClassifiers[i].predict(datasets[i].train)
            #obtain the indices of the samples that have the requirement satisfied (truly in the dataset)
            real_values_single_req = datasets[i].labels_train
            
            if(i == 0):
                final = output
                real_values = real_values_single_req
            else:
                final *= final
                real_values *= real_values_single_req

        negatively_classified = np.where(final == 0)[0]
        true_negative = np.where(real_values == 0)[0]

        print(f"Number of samples with not all requirements satisfied (according to model): {negatively_classified.shape[0]}")
        print(f"Number of samples with not all requirements satisfied (real data): {true_negative.shape[0]}")
        #calculate false negatives
        f_n = negatively_classified.shape[0]- np.intersect1d(true_negative, negatively_classified).shape[0]
        print(f"Number of false negatives from model: {f_n}")
        #calculate the missclassified real negative
        m_r_n = true_negative.shape[0] - np.intersect1d(true_negative, negatively_classified).shape[0]
        print(f"Number of missclassified real negatives: {m_r_n}")

        explanations = []

        for j, p_sample in enumerate(positively_classified):
            intersected_exp = {}
            for i in range(req_number):
                #get the sample
                sample = datasets[i].train[p_sample]
                #explain the sample
                exp = explainer[i].explain_instance(sample, self.reqClassifiers[i].predict, threshold=0.95)
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
        
        missing = 0
        explanations_reordered = []
        for exp in explanations:
            exp_reordered = {}
            for k in feature_names:
                if k in exp:
                    exp_reordered[k] = exp[k]
                else:
                    exp_reordered[k] = (-inf, inf, False, False)
                    print(k, "missing, added: ", exp_reordered[k])
                    index = explanations.index(exp)
                    missing = 1
            if missing:
                print(exp_reordered)
                missing = 0
            explanations_reordered.append(exp_reordered)
            self.explanations = explanations_reordered
        
        #self.negative_explanations = self.create_negative_anchors(negatively_classified, datasets, self.reqClassifiers, req_number, explainer)

    def process_positive_sample(self,j, positively_classified, datasets, models, req_number):
        p_sample = positively_classified[j]
        intersected_exp = {}

        for i in range(req_number):
            sample = datasets[i].train[p_sample]
            exp = self.explainers[i].explain_instance(sample, models[i].predict, threshold=0.95)
            exp_names = exp.names()
            for boundings in exp_names:
                quoted, rest = self.__get_anchor(boundings)
                parsed = self.__parse_range(rest)
                if quoted not in intersected_exp:
                    intersected_exp[quoted] = parsed
                else:
                    intersected_exp[quoted] = self.__intersect(intersected_exp[quoted], parsed)

        return intersected_exp

    def create_negative_anchors(self, negatively_classified, datasets, models, req_number, explainer):
        print("Starting negative explanations creation...")
        explanations = []

        for j, p_sample in enumerate(negatively_classified):
            intersected_exp = {}
            for i in range(req_number):
                #get the sample
                sample = datasets[i].train[p_sample]
                #explain the sample
                exp = explainer[i].explain_instance(sample, self.reqClassifiers[i].predict, threshold=0.95)
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
        
        missing = 0
        explanations_reordered = []
        for exp in explanations:
            exp_reordered = {}
            for k in self.feature_names:
                if k in exp:
                    exp_reordered[k] = exp[k]
                else:
                    exp_reordered[k] = (-inf, inf, False, False)
                    print(k, "missing, added: ", exp_reordered[k])
                    index = explanations.index(exp)
                    missing = 1
            if missing:
                print(exp_reordered)
                missing = 0
            explanations_reordered.append(exp_reordered)
            self.explanations = explanations_reordered
        
        return explanations_reordered
    
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
                            out[i] = 0 #If the sample is not inside the anchor for some feature, we set the output to 0
                            break
                if flag:
                    break
                else:
                    flag = True
            #For each sample out will be 1 if it is inside the anchor for all features, 0 otherwise
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
        """
            Generates a grid of points within the specified ranges for each feature, classifies 
            these points, and filters out points classified with a label of 1 (which means that they are already inside anchors).
            This way we get new points that have never been seen before.

            Parameters:
            -----------
            explanations : any
            feature_names : list of str
                Names of the features corresponding to each dimension in the grid.
            delta : float or int
                Step size between consecutive grid points along each dimension.
            STEP : float or int
                Starting offset added to the minimum value when generating the grid points.
            min_vals : list or array-like
                Minimum values for each feature dimension; defines the lower bound of the grid.
            max_vals : list or array-like
                Maximum values for each feature dimension; defines the upper bound of the grid.

            Returns:
            --------
            grid_points : numpy.ndarray
                Array of grid points not classified as 1, where each row corresponds to a point 
                and columns correspond to feature values.

            Description:
            ------------
            1. For each feature, generate a range of values starting at `min_val + STEP` and 
            ending before `max_val`, with intervals of size `delta`.
            2. Create a meshgrid from these ranges to form a grid of points in the feature space.
            3. Flatten and stack the meshgrid to get an array of points.
            4. Classify each point using the `classify` method, which returns labels.
            5. Remove points that are classified with label 1 (already inside anchors).
            6. Return the filtered grid points.
        """
        variables = []
        for i in range(len(feature_names)):
            variables.append(np.arange(int(min_vals[i]+STEP), int(max_vals[i]), int(delta)))
        grid_points = np.meshgrid(*variables, indexing='ij')
        grid_points = np.stack([g.ravel() for g in grid_points], axis=-1)
        out = self.classify(grid_points, explanations, feature_names)
        
        idx_del = np.where(out == 1)[0]
        grid_points = np.delete(grid_points, idx_del, axis=0)
        return grid_points
    

    def __evaluate_grid_points(self, grid_points, models, positive_samples, req_names, min_idx_cf= 0, max_idx_cf = 3):
        """
            Evaluates grid points by merging them with segments of positive samples, classifying 
            the merged points with multiple models, and filtering based on model predictions.

            Parameters:
            -----------
            grid_points : array-like, shape (n_grid_points, n_NCF)
                Grid points only containing non controllable features to be merged and evaluated.
            models : list
                List of trained models, each with a `predict` method, corresponding to different requirements.
            positive_samples : array-like, shape (n_positive_samples, n_features)
                Samples classified as positive, used to merge with grid points.
            req_names : list of str
                Names of the requirements or models corresponding to each model in `models`.
            min_idx_cf : int, optional, default=0
                Starting index for slicing the positive sample to merge with grid points.
            max_idx_cf : int, optional, default=3
                Ending index for slicing the positive sample to merge with grid points.

            Returns:
            --------
            positive_merged_points : numpy.ndarray
                Array of merged points that are positively classified (non-zero) by the combined model predictions. They are points that are not inside anchors for the NCF 
                but that have been classified as positive by the model, making them perfect to generate new anchors on.

            Description:
            ------------
            1. For each grid point and each positive sample:
                - Extract a slice from the positive sample (`min_idx_cf:max_idx_cf`).
                - Concatenate this slice with the grid point features.
                - Append the last element of the positive sample to the merged point.
                - Collect all merged points.
            2. For each model:
                - Predict labels for all merged points.
                - Combine predictions across models by element-wise multiplication.
            3. Identify merged points where the combined prediction is non-zero (positive).
            4. Return these positively classified merged points.
        """
        merged_points = []
        for grid_point in grid_points:
            for point in positive_samples:
                #print("point:", point.shape)
                #print("grid_point:", grid_point.shape)
                merged_point = np.concatenate((point[min_idx_cf:max_idx_cf], grid_point))
                merged_point = np.concatenate((merged_point, [point[-1]]))
                merged_points.append(merged_point)
                #print("merged_point:", merged_point.shape)
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
        """
            Generates new anchor explanations by intersecting feature intervals from explanations 
            of positively classified samples across multiple requirements/models.

            Parameters:
            -----------
            datasets : list
                List of dataset objects, each containing training data (currently unused but available).
            explainers : list
                List of explainer objects, each providing an `explain_instance` method to explain samples.
            models_positives : list or array-like
                List or array of samples that have been positively classified by the models.
            models : list
                List of models corresponding to each requirement, each with a `predict` method.
            req_number : int
                Number of requirements to consider.
            explanations : list of dict
                List of existing explanations (anchors) to be appended with new ones.

            Returns:
            --------
            explanations : list of dict
                Updated list of explanations where each explanation is a dictionary mapping feature 
                names to their respective intervals (anchors). Missing features are filled with 
                default intervals (-inf, inf, False, False).

            Description:
            ------------
            For each positively classified sample:
            1. For each requirement/model:
                - Explain the sample using the corresponding explainer and model.
                - Extract textual explanations from the explainer.
                - Parse these explanations into feature intervals using helper methods (`__get_anchor`, `__parse_range`).
                - Intersect these intervals across requirements to find common anchors.
            2. Append the intersected explanation to the explanations list.

            Then, for all explanations:
            - Reorder and fill missing features based on a predefined list of feature names.
            - Missing features are assigned a default interval of (-inf, inf, False, False).
            - Print warnings if features are missing in any explanation.
            """
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
        
        #TODO: feature names da passare come parametro
        feature_names = ['cruise speed','image resolution','illuminance','controls responsiveness','power','smoke intensity','obstacle size','obstacle distance','firm obstacle']
        missing = 0
        explanations_reordered = []
        for exp in explanations:
            exp_reordered = {}
            for k in feature_names:
                if k in exp:
                    exp_reordered[k] = exp[k]
                else:
                    exp_reordered[k] = (-inf, inf, False, False)
                    print(k, "missing, added: ", exp_reordered[k])
                    index = explanations.index(exp)
                    missing = 1
            if missing:
                print(exp_reordered)
                missing = 0
            explanations_reordered.append(exp_reordered)
        explanations = explanations_reordered    

        return explanations
    
    def augment_coverage(self, datasets, explanations, feature_names, min_vals, max_vals, models, positive_samples, req_names, explainers, req_number, min_idx_cf= 0, max_idx_cf = 3, delta = 40, STEP = 20, threshold = 0.5):
        """
            Incrementally augments the coverage of given explanations by generating new anchors 
            from grid points sampled within feature ranges until a specified coverage threshold is reached.

            Parameters:
            -----------
            datasets : list
                List of dataset objects used for generating new explanations.
            explanation : list of dict
                Current list of explanations (anchors), each a mapping from feature names to intervals.
            feature_names : list of str
                Names of features defining the explanation space.
            min_vals : list or array-like
                Minimum values for each feature dimension, defining the lower bound for grid sampling.
            max_vals : list or array-like
                Maximum values for each feature dimension, defining the upper bound for grid sampling.
            models : list
                List of models to evaluate the generated grid points.
            positive_samples : array-like
                Samples that are positively classified by the models, used for merging with grid points.
            req_names : list of str
                Requirement names corresponding to each model.
            explainers : list
                List of explainer objects for generating explanations of samples.
            req_number : int
                Number of requirements/models to consider.
            min_idx_cf : int, optional (default=0)
                Start index for slicing positive samples during merging with grid points.
            max_idx_cf : int, optional (default=3)
                End index for slicing positive samples during merging with grid points.
            delta : int or float, optional (default=20)
                Step size for generating grid points along each feature dimension.
            STEP : int or float, optional (default=20)
                Initial offset for grid point generation; incremented on each iteration.
            threshold : float, optional (default=0.5)
                Target coverage threshold to reach before stopping augmentation.

            Returns:
            --------
            explanation : list of dict
                Updated list of explanations after augmenting coverage with new anchors.

            Description:
            ------------
            1. Compute the initial coverage of the current explanations.
            2. While coverage is below the threshold:
            - Generate grid points within the feature space, offset by STEP and spaced by delta.
            - Evaluate these grid points combined with positive samples against the models.
            - Create new anchor explanations based on positively classified points.
            - Update the coverage measurement.
            - Increase STEP to explore new regions in subsequent iterations.
            3. Stop when coverage meets or exceeds the threshold, or when no new grid points are found.
            """
        coverage = self.coverage(explanations, feature_names) #This gives us the initial coverage from the current explanations
        #Iterate until the coverage is greater than the threshold
        while(coverage < threshold):
            grid_points = self.__grid_points_in_RN(explanations, feature_names, delta, STEP, min_vals, max_vals)
            print("grid_points:", grid_points.shape)
            if grid_points.shape[0] == 0:
                print("No grid points found, stopping augmentation.")
                break
            model_positives = self.__evaluate_grid_points(grid_points, models, positive_samples, req_names, min_idx_cf, max_idx_cf)
            print("model_positives:", model_positives.shape)
            explanations = self.__create_new_anchors(datasets, explainers, model_positives, models, req_number, explanations)
            coverage = self.coverage(explanations, feature_names)
            STEP += STEP
        return explanations
    
    def min_dist_polytope(self, x, explanations_table, controllable_features, observable_features): #feature names must be only controllable features if observable features are "satisfied" otherwise
        """
        Computes the minimum squared Euclidean distance from a given point `x` to a set of 
        polytopes defined by feature intervals for controllable and observable features.

        The distance is measured separately for controllable and observable features by 
        calculating how far `x` lies outside the interval bounds of each polytope. If `x` 
        lies inside the interval for a feature, that feature contributes zero to the distance.

        Parameters:
        -----------
        x : array-like, shape (n_features,)
            Input point for which distances to polytopes are calculated.
            The order of features in `x` is assumed to be [controllable features..., observable features...].
        explanations_table : list of dicts
            Each dict maps feature names to intervals representing a polytope.
            Intervals are tuples/lists in the form (lower_bound, upper_bound, ...).
        controllable_features : list of str
            List of controllable feature names whose intervals are checked against the 
            first part of `x`.
        observable_features : list of str
            List of observable feature names whose intervals are checked against the 
            latter part of `x`. Assumed to start after controllable features in `x`.

        Returns:
        --------
        contr_f_dist : numpy.ndarray
            Array of squared distances between `x` and each polytope considering controllable features only.
        obs_f_dist : numpy.ndarray
            Array of squared distances between `x` and each polytope considering observable features only.
        min_dist_controllable : float
            Minimum squared distance among all polytopes for controllable features.
        min_dist_index_controllable : int
            Index of the polytope with the minimum controllable feature distance.
        min_dist_observable : float
            Minimum squared distance among all polytopes for observable features.
        min_dist_index_observable : int
            Index of the polytope with the minimum observable feature distance.

        Description:
        ------------
        - For each polytope (i.e., each explanation in `explanations_table`), this function:
        1. Iterates over controllable features:
            - If `x`'s value for the feature is outside the polytope interval, 
                adds squared distance from the nearest bound.
            - Stops early if the distance for the current polytope exceeds the minimum found so far.
        2. Repeats the same process for observable features, adjusting indices accordingly.
        - After evaluating all polytopes, finds and returns the minimum distances and their indices.
        - If there are polytopes with zero distance for both controllable and observable features,
        the function favors the polytope common to both sets.

        Notes:
        ------
        - Bounds of `-inf` and `inf` in intervals are replaced by 0 and 100 respectively for distance calculation.
        - The observable features are assumed to start at index offset `len(controllable_features)` in `x`.
        """
        #print("x: ", x)
        min_dist_controllable = np.inf
        min_dist_index_controllable = -1

        min_dist_observable = np.inf
        min_dist_index_observable = -1

        contr_f_dist = np.zeros(len(explanations_table))
        obs_f_dist = np.zeros(len(explanations_table))

        #print("explanations_table lenght: ", len(explanations_table))
        for i in range(len(explanations_table)):
            #print("i: ", i)
            for j, f_name in enumerate(controllable_features):
                a, b = explanations_table[i][f_name][0], explanations_table[i][f_name][1]
                #TODO: METTERE QUESTO NELLE FUNZIONI OFFLINE
                if a == -inf:
                    a = 0
                if b == inf:
                    b = 100
                #print("a: ", a)
                #print("b: ", b)
                #print("x[j]: ", x[j])
                if(x[j] < a):
                    d = (a - x[j]) ** 2
                    contr_f_dist[i] += d
                    #print("contr_f_dist[i]: ", contr_f_dist[i])
                    if(contr_f_dist[i] >= min_dist_controllable):
                        #print("distanza maggiore della minima controllable")
                        break
                elif(x[j] > b):
                    d = (b - x[j]) ** 2
                    contr_f_dist[i] += d
                    #print("contr_f_dist[i]: ", contr_f_dist[i])
                    if(contr_f_dist[i] >= min_dist_controllable):
                        #print("distanza maggiore della minima controllable")
                        break
                #elif a <= x[j] <= b:
                #    dist[i] += 0
            if(contr_f_dist[i] < min_dist_controllable):
                min_dist_controllable = contr_f_dist[i]
                min_dist_index_controllable = i
                #print("min_dist_controllable: ", min_dist_controllable)
                #print("min_dist_index_controllable: ", min_dist_index_controllable)
            #print("finito controllable features")
            #print("contr_f_dist: ", contr_f_dist)


            for j, f_name in enumerate(observable_features): 
                #NCF start from 3 not from 0, therefor we need to add 3 to the index
                jj = j + 3
                #print("f_name", f_name,explanations_table[i])
                a, b = explanations_table[i][f_name][0], explanations_table[i][f_name][1]
                if a == -inf:
                    a = 0
                if b == inf:
                    b = 100
                #print("a obs: ", a)
                #print("b obs: ", b)
                #print("x[j] obs: ", x[jj])

                if(x[jj] < a):
                    d = (a - x[jj]) ** 2
                    #dist[i] += d
                    obs_f_dist[i] += d
                    #print("obs_f_dist[i]: ", obs_f_dist[i])
                    if(obs_f_dist[i] >= min_dist_observable):
                        #print("distanza maggiore della minima observable")
                        break
                elif(x[jj] > b):
                    d = (b - x[jj]) ** 2
                    #dist[i] += d
                    obs_f_dist[i] += d
                    #print("obs_f_dist[i]: ", obs_f_dist[i])
                    if(obs_f_dist[i] >= min_dist_observable):
                        #print("distanza maggiore della minima observable")
                        break
                #elif a <= x[j] <= b:
                #    dist[i] += 0
            if(obs_f_dist[i] < min_dist_observable):
                min_dist_observable = obs_f_dist[i]
                min_dist_index_observable = i
                #print("min_dist_observable: ", min_dist_observable)
                #print("min_dist_index_observable: ", min_dist_index_observable)
            #print("finito observable features")
            #print("obs_f_dist: ", obs_f_dist)

        #This adds the case in which there are muptile polytopes with the same distance, choose the one common to both
        mask_controllable = np.where(contr_f_dist == 0)[0]
        mask_observable = np.where(obs_f_dist == 0)[0]
        for index in mask_observable:
            if index in mask_controllable:
                min_dist_index_observable = index
                min_dist_index_controllable = index
                break

        return contr_f_dist, obs_f_dist, min_dist_controllable, min_dist_index_controllable, min_dist_observable, min_dist_index_observable


    def evaluate_sample(self, sample, threshold = 0.8):
        """
        Evaluates a given sample against a set of polytopes (explanations) and classification models,
        considering both controllable and observable features.

        The function first computes the minimum squared distances from the sample to each polytope for
        controllable and observable features separately. If the sample lies inside the polytope for observable
        features, it checks if it is also inside for controllable features:
        - If inside both, it directly evaluates the sample using the models.
        - If inside observable but outside controllable, it modifies the sample to move it inside the
            controllable polytope bounds, then re-evaluates.
        If the sample is outside observable polytopes, it still evaluates the sample with the models and
        returns the results along with the distance to the closest observable polytope.

        Parameters:
        -----------
        sample : array-like, shape (n_features,)
            The input sample to be evaluated. Features should be ordered as controllable features followed by observable features.
        explanation : list of dicts
            List of polytopes, where each polytope is a dictionary mapping feature names to intervals.
        controllable_features : list of str
            List of controllable feature names.
        observable_features : list of str
            List of observable feature names.
        req_names : list of str
            List of requirement names corresponding to each model.
        models : list of model objects
            List of models corresponding to requirements, each implementing a `predict` method.

        Returns:
        --------
        min_dist_controllable : float
            Minimum squared distance from the sample to the closest polytope in controllable features.
        min_dist_observable : float
            Minimum squared distance from the sample to the closest polytope in observable features.
        sample : array-like
            Possibly adjusted sample that lies inside the polytope for controllable features if it was originally outside.
        outputs : numpy.ndarray
            Array of model predictions for each requirement after evaluation.

        Notes:
        ------
        - The method assumes that the observable features start after the controllable features in the sample vector.
        - If the sample is outside the controllable polytope but inside the observable polytope, the sample is adjusted
        to move inside the controllable polytope using `go_inside_CF_given_polytope`.
        """
        #print("Sample: ", sample)
        contr_f_dist, obs_f_dist, min_dist_controllable, min_dist_index_controllable, min_dist_observable, min_dist_index_observable = self.min_dist_polytope(sample, self.explanations, self.controllableFeaturesNames, self.observableFeaturesNames)
        # print("min_dist_controllable: ", min_dist_controllable)
        # print("min_dist_index_controllable: ", min_dist_index_controllable)
        # print("min_dist_observable: ", min_dist_observable)
        # print("min_dist_index_observable: ", min_dist_index_observable)

        if obs_f_dist[min_dist_index_observable] == 0:
            #print("The sample is in the polytope for the observable features!")
            if contr_f_dist[min_dist_index_observable] == 0:
                #print("The sample is in the polytope for the controllable features too!")
                #Evaluate the sample with the model
                outputs = np.zeros(len(self.reqNames))
                output = True
                for r, req in enumerate(self.reqNames):
                    #print(f"___________Requirement {req}___________")
                    
                    #classify the samplsses with the model
                    tmp_output = self.reqClassifiers[r].predict(sample.reshape(1, -1))
                    #print("tmp_output: ", tmp_output)
                    outputs[r] = tmp_output
                    #print("outputs: ", outputs)
                    output = output and bool(tmp_output)
                #print("output: ", output)
                
                confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))
                min_prob = np.min(confidence)
                if min_prob < threshold:
                    sample = self.findBestAdaptation(sample, self.explanations[min_dist_index_observable], self.controllableFeaturesNames)
                confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))

                return sample, confidence, outputs   
            else:
                #print("We are inside polytiope ", min_dist_index_observable, " for the observable features but not for the controllable ones, we will go there")
                polytope = self.explanations[min_dist_index_observable]
                #print("Polytope: ", polytope)

                sample = self.go_inside_CF_given_polytope(sample, polytope, self.controllableFeaturesNames, self.observableFeaturesNames)
                #print("Sample after going inside: ", sample)
                #check is its now inside the polytope
                for i, f_name in enumerate(self.controllableFeaturesNames):
                    inside = self.__inside(sample[i], polytope[f_name])
                    if not inside:
                        #print("sample not inside for feature: ", f_name)
                        inside = False
                for i, f_name in enumerate(self.observableFeaturesNames):
                    inside = self.__inside(sample[i+3], polytope[f_name])
                    if not inside:
                        #print("sample not inside for feature: ", f_name)
                        inside = False
                #if inside:
                    #print("The sample is now inside the polytope for the controllable features too!")
                
                #Evaluate the sample with the model
                outputs = np.zeros(len(self.reqNames))
                output = True
                for r, req in enumerate(self.reqNames):
                    #print(f"___________Requirement {req}___________")
                    
                    #classify the samples with the model
                    tmp_output = self.reqClassifiers[r].predict(sample.reshape(1, -1))
                    #print("tmp_output: ", tmp_output)
                    outputs[r] = tmp_output
                    output = output and bool(tmp_output)
                #print("output: ", output)
                contr_f_dist, obs_f_dist, min_dist_controllable, min_dist_index_controllable, min_dist_observable, min_dist_index_observable = self.min_dist_polytope(sample, self.explanations, self.controllableFeaturesNames, self.observableFeaturesNames)
                confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))
                min_prob = np.min(confidence)
                if min_prob < threshold:
                    sample = self.findBestAdaptation(sample, self.explanations[min_dist_index_observable], self.controllableFeaturesNames)
                confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))
                return sample, confidence, outputs   
        else:
            #print("The sample is not in the polytope for the observable features!")
            #print("The closest polytope for NCF is at dist: ",obs_f_dist[min_dist_index_observable])
            #Now we want to change the CF to get as close as possible to that polytope
            polytope = self.explanations[min_dist_index_observable]
            #print("Polytope: ", polytope)

            sample = self.go_inside_CF_given_polytope(sample, polytope, self.controllableFeaturesNames, self.observableFeaturesNames)
            #print("Sample after going closer: ", sample)

            #check is its now inside the polytope for the CF
            for i, f_name in enumerate(self.controllableFeaturesNames):
                inside = self.__inside(sample[i], polytope[f_name])
                if not inside:
                    #print("sample not inside for feature: ", f_name)
                    inside = False
            for i, f_name in enumerate(self.observableFeaturesNames):
                inside = self.__inside(sample[i+3], polytope[f_name])
                if not inside:
                    #print("sample not inside for feature: ", f_name)
                    inside = False
            #if inside:
                #print("The sample is now inside the polytope for the controllable features!")
            #Evaluate the sample with the model
            
            outputs = np.zeros(len(self.reqNames))
            output = True
            for r, req in enumerate(self.reqNames):
                #print(f"___________Requirement {req}___________")
                
                #classify the samplsses with the model
                tmp_output = self.reqClassifiers[r].predict(sample.reshape(1, -1))
                #print("tmp_output: ", tmp_output)
                outputs[r] = tmp_output
                output = output and bool(tmp_output)
            confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))
            min_prob = np.min(confidence)
            if min_prob < threshold:
                sample = self.findBestAdaptation(sample, self.explanations[min_dist_index_observable], self.controllableFeaturesNames)
            confidence =  vecPredictProba(self.reqClassifiers, sample.reshape(1, -1))
            return sample, confidence, outputs           
            

    def go_inside_CF_given_polytope(self, sample, polytope, controllable_features, observable_features):
        """
            Adjusts the controllable feature values of a sample to ensure it lies within the bounds of a given polytope.

            Parameters:
            -----------
            sample : array-like, shape (n_controllable_features + n_observable_features,)
                The input sample whose controllable features will be adjusted.
                The order of features should correspond to controllable features followed by observable features.
            polytope : dict
                A dictionary mapping feature names to intervals, representing the polytope.
                Each interval is expected as a tuple or list: (lower_bound, upper_bound, ...).
            controllable_features : list of str
                List of feature names considered controllable. Only these features are adjusted.
            observable_features : list of str
                List of observable feature names (not used in this function but kept for API consistency).

            Returns:
            --------
            sample : array-like
                The modified sample where controllable feature values are adjusted to fall within 
                the polytope bounds. If a feature value is outside the bounds, it is moved to 
                one unit inside the closest bound.

            Description:
            ------------
            - For each controllable feature:
                - If the feature value in `sample` is less than the lower bound (with -inf replaced by 0),
                set it to one unit above the lower bound.
                - If the feature value is greater than the upper bound (with inf replaced by 100),
                set it to one unit below the upper bound.
                - Otherwise, the feature value remains unchanged.
            - Observable features are not modified in this method.
            """
        for i, f_name in enumerate(controllable_features):
            a, b = polytope[f_name][0], polytope[f_name][1]
            if a == -inf:
                a = 0
            if b == inf:
                b = 100
            # print("a: ", a)
            # print("b: ", b)
            # print("x[j]: ", sample[i])
            if(sample[i] < int(a)):
                sample[i] = int(a)+1
            elif(sample[i] > int(b)):
                sample[i] = int(b)-1
            #sample[i] = (a+b)/2
        return sample


    def findBestAdaptation(self, sample, polytope, controllable_features, threshold=0.8, max_iter=1000):
        delta_controllable_features = []
        #print("sample: ", sample)
        for i, f_name in enumerate(controllable_features):
            a, b = polytope[f_name][0], polytope[f_name][1]
            if a == -inf:
                a = 0
            if b == inf:
                b = 100
            if (b-sample[i]) < (sample[i]-a):
                delta = - (b-a) * 0.1 #We need to move backwards
            else:
                delta = (b-a) * 0.1 #We need to move forward
            delta_controllable_features.append(delta)
        #print("delta_controllable_features: ", delta_controllable_features)
        #print("Starting to find the best adaptation for the sample: ", sample)
        n_iter = 0
        min_prob = 0
        best_avg_prob = 0
        old_best_avg_prob = best_avg_prob
        current_avg_prob = 0
        early_stopping_condition_counter = 0
        adapted_sample = sample
        outOfBounds_feature_counter = 0
        best_sample_iteration = 0

        #Paramteres added in order not to stall after we find the best sample
        best_current_adaptation = adapted_sample.copy()
        best_current_adaptation_avg_prob = 0

        while n_iter < max_iter and min_prob < threshold:
            for i, f_name in enumerate(controllable_features):
                #print("Moving on feature: ", f_name, " with delta: ", delta_controllable_features[i])
                adapted_sample[i] += delta_controllable_features[i]
                #print("adapted_sample: ", adapted_sample)
                n_iter += 1
                #print("Boundaries for feature ", f_name, ": ", polytope[f_name])
                if adapted_sample[i] >= min(polytope[f_name][1], 100) or adapted_sample[i] <= max(0,polytope[f_name][0]): #If we are outside the bounds of the polytope, we need to stop
                    #print("Adapted sample is outside the bounds of the polytope, stopping the search for the best adaptation for feature, bounds: ", polytope[f_name])
                    adapted_sample[i] -= delta_controllable_features[i]
                    outOfBounds_feature_counter += 1
                else:
                    outOfBounds_feature_counter = 0
                    probs = vecPredictProba(self.reqClassifiers, adapted_sample.reshape(1, -1))
                    probs = np.minimum(probs, threshold)
                    current_avg_prob = np.mean(probs)
                    #print("current_avg_prob: ", current_avg_prob, " for adapted_sample: ", adapted_sample)
                    if current_avg_prob >= best_avg_prob: #If the currect minimum probability is higher than the maximum minimum probability found so far
                        #NB: It needs to be >= or else we get stuck at doing the same loop over and over again
                        #print("New best sample found: ", adapted_sample, " with highest avg probability: ", current_avg_prob)
                        best_avg_prob = current_avg_prob
                        min_prob = np.min(probs)
                        best_sample = adapted_sample.copy()
                        best_sample_iteration = n_iter
                        #print("We got a new best sample: ", best_sample, " with avg probability: ", best_avg_prob, " and min probability: ", min_prob)

                    if current_avg_prob >= best_current_adaptation_avg_prob:
                        best_current_adaptation_avg_prob = current_avg_prob
                        best_current_adaptation = adapted_sample.copy()
                        #print("New best current adaptation found: ", best_current_adaptation, " with avg probability: ", best_current_adaptation_avg_prob)
                    adapted_sample[i] -= delta_controllable_features[i]

            
            if outOfBounds_feature_counter == len(controllable_features):
                #print("All features are out of bounds, stopping the search for the best adaptation.")
                break
            #print(n_iter, " - Best sample so far: ", best_sample, " with highest avg probability: ", best_avg_prob, "at iteration: ", best_sample_iteration)
            old_best_avg_prob = best_avg_prob
            best_current_adaptation_avg_prob = 0

            if best_avg_prob == old_best_avg_prob:
                early_stopping_condition_counter += 1
                #print("Early stopping condition counter: ", early_stopping_condition_counter)
                adapted_sample = best_current_adaptation.copy() #Reset the adapted sample to the best current adaptation found so far
                #We do this or else it will just keep looping the same adaptation
            else:
                early_stopping_condition_counter = 0
                adapted_sample = best_sample.copy() #Reset the adapted sample to the best sample found so far
            if early_stopping_condition_counter >= 50:
                #print("Early stopping condition met, stopping the search for the best adaptation.")
                break
        
        return best_sample

    def findBestAdaptationUsingNegatives(self, sample, polytope, controllable_features, threshold=0.8, max_iter=100):
        delta_controllable_features = []
        #print("sample: ", sample)
        for i, f_name in enumerate(controllable_features):
            a, b = polytope[f_name][0], polytope[f_name][1]
            if a == -inf:
                a = 0
            if b == inf:
                b = 100
            if (b-sample[i]) < (sample[i]-a):
                delta = - (b-a) * 0.1 #We need to move backwards
            else:
                delta = (b-a) * 0.1 #We need to move forward
            delta_controllable_features.append(delta)
        #print("delta_controllable_features: ", delta_controllable_features)
        #print("Starting to find the best adaptation for the sample: ", sample)
        n_iter = 0
        max_prob = 0
        current_avg_prob = 0
        early_stopping_condition_counter = 0
        adapted_sample = sample
        while n_iter < max_iter and max_prob < threshold:
            for i, f_name in enumerate(controllable_features):
                #print("Moving on feature: ", f_name, " with delta: ", delta_controllable_features[i])
                starting_value = adapted_sample[i]
                adapted_sample[i] += delta_controllable_features[i]
                #print("adapted_sample: ", adapted_sample)
                if adapted_sample[i] >= min(100,polytope[f_name][1]) or adapted_sample[i] <= max(0,polytope[f_name][0]): #If we are outside the bounds of the polytope, we need to stop
                    break
                adapted_sample_array = np.array(adapted_sample).reshape(1, -1)
                in_negatives = self.classify(adapted_sample_array, self.negative_explanations, self.feature_names)
                while in_negatives == 1:
                    #print("Adapted sample is in the negatives, trying to adapt it further")
                    adapted_sample[i] += delta_controllable_features[i]
                    print("adapted_sample: ", adapted_sample)
                    if adapted_sample[i] >= min(100,polytope[f_name][1]) or adapted_sample[i] <= max(0,polytope[f_name][0]):
                        print("Adapted sample is outside the bounds of the polytope, stopping the search for the best adaptation for feature: ", f_name)
                        break
                    adapted_sample_array = np.array(adapted_sample).reshape(1, -1)
                    in_negatives = self.classify(adapted_sample_array, self.negative_explanations, self.feature_names)
                    print("in_negatives: ", in_negatives)
                probs = vecPredictProba(self.reqClassifiers, adapted_sample.reshape(1, -1))
                
                probs = np.minimum(probs, threshold)
                current_avg_prob = np.mean(probs)

                if current_avg_prob >= max_prob: #If the currect minimum probability is higher than the maximum minimum probability found so far
                    #NB: It needs to be >= or else we get stuck at doing the same loop over and over again
                    #print("New best sample found: ", adapted_sample, " with highest minimum probability: ", current_max_prob)
                    max_prob = current_avg_prob
                    best_sample = adapted_sample.copy()
                #print(n_iter, " - Best sample so far: ", best_sample, " with highest minimum probability: ", max_prob)
                n_iter += 1
                adapted_sample[i] = starting_value
            #print(n_iter, " - Best sample so far: ", best_sample, " with highest minimum probability: ", max_prob)
            if current_avg_prob == max_prob:
                early_stopping_condition_counter += 1
            if early_stopping_condition_counter >= 50:
                #print("Early stopping condition met, stopping the search for the best adaptation.")
                break
            adapted_sample = best_sample.copy() #Reset the adapted sample to the best sample found so far
        
        return best_sample