from typing import Optional, Tuple, List, Dict
import numpy as np


class PreprocessorBase:
    """
    Basic preprocessor class
    """

    def __init__(self) -> None:
        """
        Init method
        """
        pass
    
    def __call__(self, arr: np.ndarray):
        """
        Applies transform.

        :param arr: Input image
        :return: Processed image
        """
        return arr


class ZscorePreprocessor(PreprocessorBase):

    def __init__(
        self, mean: List[float], std: List[float]
    ) -> None:
        """
        Z-score preprocessing class

        :param mean: List with channel means in train set
        :param std: List with channel stds in train set
        """
        self.mean = np.asarray(mean)
        self.std = np.asarray(std)
    
    def __call__(self, arr: np.ndarray):
        """
        Applies transform.

        :param arr: Input image
        :return: Processed image
        """
        return (arr - self.mean) / self.std


class MinMaxPreprocessor(PreprocessorBase):
    
    def __init__(self) -> None:
        """
        Min-max preprocessing class (0-1 normalization).
        """
        pass
    
    def __call__(self, arr: np.ndarray):
        """
        Applies transform.

        :param arr: Input image
        :return: Processed image
        """
        minVal = arr.min()
        return (arr - minVal) / (arr.max() - minVal)