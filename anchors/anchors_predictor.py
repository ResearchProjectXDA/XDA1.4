import numpy as np
import re
from math import inf


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

    def __get_anchor(a)-> tuple:
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

    
    def __parse_range(expr: str):
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

    def __inside(val, interval) -> bool:
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

        