# -*- coding: utf-8 -*-
"""
This module consists of network class definitions, used to hide implementation
details of used deep learning libraries.
"""
from abc import ABC, abstractmethod

import numpy as np


class Network(ABC):
    """
    This class hides the functional API of (e.g.) Keras models and reduces the
    available features to the bare minimum. This eases adapting to other deep
    learning libraries.
    """
    @abstractmethod
    def predict(self, x):#pylint: disable=C0103
        """
        Parameters
        ----------
        x : array_like
            The input (batch of inputs) which should be processed by this
            network.

        Returns
        -------
        y : array_like
            The predicted class confidence(s) belonging to input `x`. This is
            the (raw) output of the network, before applying argmax -- but
            (depending on the network's topology) after softmax.
        """
        raise NotImplementedError

    def labels(self, x):
        """
        Parameters
        ----------
        x : array_like
            The input (batch of inputs) which should be processed by this
            network.

        Returns
        -------
        labels : array_like of int
            The predicted class labels (determined by numpy's argmax function)
            belonging to `x`.
        """
        predictions = self.predict(x)
        labels = np.argmax(predictions, axis = 1)
        return labels

    @staticmethod
    @abstractmethod
    def bounds():
        """
        Returns
        -------
        bounds : (float, float)
            Lower and upper bound of input scalars (e. g. pixel values).
            The default implementation assumes the network to be trained on the
            channel centered ImageNet training data set, using BGR encoding.

        Notes
        -----
        Although this methods provides a default implementation, it is tagged as
        abstractmethod, to force inheriting classes to define their bounds
        explicitly.
        """
        bgr_mean_pixel = [103.939, 116.779, 123.68]
        bnds = (np.subtract(0, max(bgr_mean_pixel), dtype = np.float32),
                np.subtract(255, min(bgr_mean_pixel), dtype = np.float32) )
        return bnds

    @abstractmethod
    def name():
        """
        Returns
        -------
        name : str
            A class-unique human readable identifier of the network.
        """

class KerasNetwork(Network):#pylint: disable=W0223
    """
    An intermediate class to make it easier to lift Keras applications to the
    `Network` interface.
    """
    def __init__(self, model):
        self._wrapped_model = model

    def predict(self, x):
        return self._wrapped_model.predict(x)

    def wrapped_model(self):
        return self._wrapped_model