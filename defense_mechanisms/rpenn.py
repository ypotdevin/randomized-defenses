# -*- coding: utf-8 -*-

import logging
from datetime import datetime

import numpy as np

from .base import DefenseMechanism


class RPENN(DefenseMechanism):
    """
    This class provides several methods which try to predict the `true` label
    (the label of the original input corresponding to the adversarial one) of
    the, in those methods, provided adversarial input. It does so by generating
    `n_variations` variations of the, in those methods, provided model. By
    `variation` is meant the following:
        Given a model m, a variation m' of m has the same structure as m, but
        the weights of m' are slightly different. 'Slightly different' means:
            For all i and j the jth weight of layer i from model m' w'_i_j is a
            sample drawn from a normal distribution with mean m_i_j (the
            corresponding weight from m) and standard deviation
            `rel_std` * m_i_j.
    The predictions of each variation are then used to determine the 'true'
    label of the adversarial input. The idea of perturbing the weights of a
    network comes from the assumption, that adversarial attacks rely on the
    specific setting of the weights to launch their attack. Slightly changing
    the weights should not hurt the overall accuracy of the network, but should
    on average deny the adversarial attack.
    """
    def __init__(
            self,
            keras_model,
            *args,#pylint: disable=W0613
            rel_std = 0.05,
            n_variations = 7,
            **kwargs):#pylint: disable=W0613
        # Have also *args and **kwargs to be able to initialize objects of this
        # class using dictionary containing also other stuff.
        self.model = keras_model
        self.rel_std = rel_std
        self.n_variations = n_variations

    def _similar_layer(self, layer):
        """
        See https://stackoverflow.com/a/49860446/3389669.
        """
        sigma = self.rel_std * layer
        expected_value = layer
        return sigma * np.random.randn(*layer.shape) + expected_value

    def _similar_weights(self, weights):
        return [self._similar_layer(layer) for layer in weights]

    def predict_n(self, *batches):
        """
        Implements `defense_mechanisms.DefenseMechanism.predict_n`.
        """
        orig_weights = self.model.get_weights()

        def log_and_predict(j, batch):
            logging.debug(
                '%s '
                'defense_mechanisms.perturb_weights.predict_n(): '
                'Predicting batch %d.',
                datetime.now().isoformat(),
                j,
            )
            current_weights = self._similar_weights(orig_weights)
            self.model.set_weights(current_weights)
            return self.model.predict(batch)

        for i in range(self.n_variations):
            logging.debug(
                '%s '
                'defense_mechanisms.perturb_weights.predict_n(): '
                'Generating variation %d.',
                datetime.now().isoformat(),
                i,
            )
            yield tuple(
                log_and_predict(j, batch) for (j, batch) in enumerate(batches)
            )

        self.model.set_weights(orig_weights)
        logging.debug(
            'defense_mechanisms.perturb_weights.predict_n(): '
            'Processed every batch. Restored original weights.'
        )

    @staticmethod
    def parameter_names():
        return ['rel_std', 'n_variations']
