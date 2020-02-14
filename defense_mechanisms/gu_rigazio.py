# -*- coding: utf-8 -*-
"""
This module contains the defense mechanism (L1 and L*) introduced by Gu and
Rigazio in https://arxiv.org/abs/1412.5068. Our adaptation L+ is also found in
the `GuRigazio` class.
"""
from math import ceil

import numpy as np

from .base import DefenseMechanism


class GuRigazio(DefenseMechanism):
    """
    This defense mechanism realizes interpretations of Gu and Rigazio's Gaussian
    noise injection proposal (not the Gaussian blurring).
    """
    def __init__(
            self,
            keras_model,
            noise_stddev,
            how,
            *args,#pylint: disable=W0613
            interpretation = None,
            **kwargs):#pylint: disable=W0613
        # Have also *args and **kwargs to be able to initialize objects of this
        # class using dictionary containing other stuff too.
        """
        Parameters
        ----------
        keras_model
        noise_stddev : float
            The standard deviation of the noise used to perturb the input layer
            (and the hidden layers, if `how` is 'L*').
        how : str
            One of 'L1', 'L*', 'L+'. 'L1' implies introducing additive noises just
            into the input layer (the input itself). 'L*' means also injecting
            the noises into the hidden layers. 'L+' means injecting noises just
            into the hidden layers.
        interpretation : None or str
            One of None, 'weights' or 'activations'. Gu and Rigazio wrote about
            'applying' noise to layers. It is not specified how the application
            is executed in detail. For example whether the application should be
            performed before calculating the activation of a layer (e.g. by
            manipulating the weights), or afterwards (perturbing the already
            calculated activation value, before it is passed to the succeeding
            layer).
        """
        self._model = keras_model
        self._sigma = noise_stddev
        self._how = how
        self._interpretation = interpretation

    def predict_n(self, *batches):
        if self._how == 'L1' and self._interpretation is None:
            yield self._predictions('L1', batches)
        elif self._how in ['L*', 'L+'] and self._interpretation == 'weights':
            orig_weights = self._inject_noise_into_weights()
            yield self._predictions(self._how, batches)
            self._restore_weights(orig_weights)
        else:
            raise ValueError(
                'Not supported combination of `how`: {} '
                'and `interpretation`: {}.'.format(
                    self._how, self._interpretation
                )
            )

    def _predictions(self, how, batches):
        batch_transformer = {
            'L1' : self._noisy_layer,
            'L*' : self._noisy_layer,
            'L+' : lambda x: x,
        }[how]
        return tuple(
            self._model.predict(batch_transformer(batch))
            for batch in batches
        )

    def _inject_noise_into_weights(self):
        """
        Inject noise into the weight layers of the model encapsuled in `self`
        (side effect).

        Returns
        -------
        The weights right before transformation (to have the opportunity to
        restore them).
        """
        orig_weights = self._model.get_weights()
        noisy_weights = self._noisy_weights(orig_weights)
        self._model.set_weights(noisy_weights)
        return orig_weights

    def _noisy_weights(self, weights):
        return [self._noisy_layer(layer) for layer in weights]

    def _noisy_layer(self, layer):
        # Since G&R use the plural term 'noises' in their paper, it is assumed
        # to draw 'fresh' noises for each component in `layer` (in contrast to
        # use one noise per layer or even for all layers).
        expected_value = layer
        np.random.seed()
        return self._sigma * np.random.randn(*layer.shape) + expected_value

    def _restore_weights(self, weights):
        self._model.set_weights(weights)

    @staticmethod
    def parameter_names():
        return ['noise_stddev', 'how', 'interpretation']
