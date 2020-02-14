# -*- coding: utf-8 -*-
"""
This module consists of classes and functions useful for defining and using
defense mechanisms.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
from sklearn import preprocessing


class DefenseMechanism(ABC):
    """
    This class defines defense mechanisms to be wrappers around a keras neural
    network model (therefore the same name `predict` as for a keras networks).
    """
    def predict(self, batch):
        """
        This method predicts at least once the classes for the given input batch.

        Parameters
        ----------
        batch : numpy.ndarray
            A batch of inputs (usually images).

        Returns
        -------
        confidences_batches : iterable of numpy.ndarray
            Each array is a batch of confidences (result of last layer of a
            network). This is an iterable (instead of just one batch) to enable
            implementing non-deterministic defence mechanisms d, which result in
            multiple votes. If d is deterministic, confidences_batches is just a
            singleton.

        Notes
        -----
        The implementing methods are required to just use O(1) memory space and
        therefore it is permitted for the iterable, to be an iterator (just
        supporting one iteration, then being exhausted).
        """
        return (confidences_batch for (confidences_batch,) in self.predict_n(batch) )

    def predict_2(self, batch0, batch1):
        """
        This is a generalization of `predict`, which infers the confidences of
        `batch0` and `batch1` "at once" – keeping the same internal state of the
        defense mechanism. Use this for example to compare the predictions for
        benign input and for adversarial input, using the same internal
        "setting" for both.
        Simplified

            `predict_n(b1, b2) = ps`

        holds, where ps is an iterable of pairs of numpy.ndarray. Each pair
        p = (e1, e2) contains predictions for b1, b2. Within each pair, the
        mentioned "internal setting" is the same. The number of tuples in the
        iterable is determined by the implementing defense mechanism.

        Parameters
        ----------
        batch0, batch1 : numpy.ndarray
            Batches of inputs (usually images).

        Returns
        -------
        ps : iterable of pair of numpy.ndarray

        Notes
        -----
        The implementing methods are required to just use O(1) memory space and
        therefore it is permitted for the iterable, to be an iterator (just
        supporting one iteration, then being exhausted).
        """
        return self.predict_n(batch0, batch1)

    @abstractmethod
    def predict_n(self, *batches):
        """
        This a generalization of `predict_2` (which is itself a generalization
        of `predict`) to more than two input batches. Like `predict_2`, it keeps
        the predictions for each batch in `batches` comparable.
        Simplified

            `predict_n(b1, b2, ..., bn) = ts`

        holds, where ts is an iterable of tuples of numpy.ndarray. Each tuple
        has n entries, each entry ei (for i = 1, ..., n) is a prediction for bi.
        Within each tuple, the predictions are comparable (regarding the
        internal state). The number of tuples in the iterable is determined by
        the implementing defense mechanism.

        Parameters
        ----------
        batches : list of numpy.ndarray
            Each array is a batch of inputs (usually images).

        Returns
        -------
        ts : iterable of tuple of numpy.ndarray
            See the explanation given at `aggregate_predict_n_by`.

        Notes
        -----
        The implementing methods are required to just use O(1) memory space and
        therefore it is permitted for the iterable, to be an iterator (just
        supporting one iteration, then being exhausted).
        """
        raise NotImplementedError

    @abstractmethod
    def parameter_names(self):
        """
        Returns
        names : list of str
            The names of the parameters essential for this defense mechanism.
            The implementation is expected to be a static method.
        """
        return NotImplementedError



def aggregate_predict_n_by(methods, predict_n_result):
    """
    If a defense mechanism is non-deterministic/randomized, it provides several
    prediction alternatives for each of the input batches b1, b2, ..., bn it was
    applied to. So dm.predict_n(b1, b2, ..., bn) equals roughly

        cc11, cc21, ..., ccn1
        cc12, cc22, ..., ccn2
                  .
                  .
                  .
        cc1m, cc2m, ..., ccnm,

    where each ccij is a class confidence vector batch (a batch of vectors,
    which contain for each class a confidence value), corresponding to batch bi
    and alternative j (out of m alternatives).

    To come to a final prediction for each input batch, one has to consider and
    combine the given alternatives. Two ways, for example, are averaging and
    picking the majority vote (which class is supported by the most
    alternatives). Forging the alternatives to a final vote is called
    aggregation in this case and that is the purpose of this function.

    For efficiency reasons the 'lines' shown here are provided by a generator
    (only one line at a time will be available). That's why all desired ways to
    aggregate the alternatives have to be performed 'simultaneously' (the reason
    `methods` being a list).

    Parameters
    ----------
    methods : list of str
        A selection of `_AVAILABLE_CONSUMERS.keys()`.
    predict_n_result : iterable of tuple of array_like
        The result of a `dm.predict_n(*batches)` call.

    Returns
    -------
    ccs : dict
        A dictionary which keys are the methods and which values are tuples of
        class confidence vector batches). So basically ccs equals
            {
                method_a : (cc1_a, cc2_a, ..., ccn_a),
                method_b : (cc1_b, cc2_b, ..., ccn_b),
                ...
            },
        where cci_* is the aggregation of cci1, cci2, ..., cci_m by method *.
    """
    if any(method not in _AVAILABLE_AGGREGATORS.keys() for method in methods):
        raise ValueError(
            'Unsupported aggregation method selection: {}'.format(methods)
        )
    predict_n_result_iter = iter(predict_n_result)

    aggregators = {}
    try:
        # Pulling out the first alternative is necessary to initialize the
        # consumer objects.
        alternative_1_batches = next(predict_n_result_iter)
        for method in methods:
            aggregators[method] = tuple(
                _AVAILABLE_AGGREGATORS[method](representative = batch)
                for batch in alternative_1_batches
            )
        _push_into_aggregators(aggregators, alternative_1_batches)
    except StopIteration:
        logging.error(
            '%s %s.%s(): '
            'Provided an empty iterable for argument `predict_n_result`.',
            datetime.now().isoformat(),
            aggregate_predict_n_by.__module__,
            aggregate_predict_n_by.__qualname__,
        )
        raise ValueError(
            'Provided an empty iterable for argument `predict_n_result`.'
        )
    # the remaining alternatives
    for batches in predict_n_result_iter:
        _push_into_aggregators(aggregators, batches)

    ccs = _aggregators_to_aggregation(aggregators)
    return ccs

def _push_into_aggregators(ccs, batches):
    for aggregator_tuple in ccs.values():
        for (aggregator, batch) in zip(aggregator_tuple, batches):
            aggregator.consume(batch)

def _aggregators_to_aggregation(aggregators):
    ccs = {}
    for (method, aggregator_tuple) in aggregators.items():
        ccs[method] = [aggr.aggregation() for aggr in aggregator_tuple]
    return ccs



class CountAggregator():
    """
    Treating

                 v1_b1, ..., v1_bn
                        ...
        votes := vi_b1, ..., vi_bn = predict_n(b1, ..., bn)
                        ...
                 vm_b1, ..., vm_bn

    as votes of an ensemble of size m predicting the classes of the batches b1
    to bn, CountAggregator will aggregate these votes by counting how many times
    a class is predicted for an entry in a batch by a member of the ensemble.
    The result of the aggregation are batches b1', ..., bn' having the same
    lengths as b1, ..., bn where b[i][j] is the number of ensembles predicting
    class j for entry i. This result may be used to perform a majority vote.
    """
    def __init__(self, representative, normalize = False):
        self._normalize = normalize
        d_type = np.float_ if normalize else np.int_
        self._counts = np.zeros_like(representative, dtype = d_type)
        self._eye = np.eye(self._counts.shape[1], dtype = d_type)

    def consume(self, array_like):
        assert self._counts.shape == array_like.shape, 'Shape mismatch: {} vs {}'.format(
            self._counts.shape, array_like.shape
        )
        self._counts += self._to_0_1_array(array_like)

    def _to_0_1_array(self, array):
        """
        For every row in the argument array: Set every entry to 0, except the
        maximum value of that row -- set that one to 1.
        """
        indices = np.argmax(array, axis = 1)
        return np.squeeze(self._eye[indices])

    def aggregation(self):
        if self._normalize:
            return preprocessing.normalize(self._counts, norm = 'l1')
        return self._counts

class MeanAggregator():
    """
    Like `CountAggregator` but instead of counting the number of times an
    ensemble member predicts a class for an input, average the vote of each
    ensemble member into one vote.
    """
    def __init__(self, representative, normalize = True):
        self._normalize = normalize
        self._mean = np.zeros_like(representative)
        self._processed = 0

    def consume(self, array_like):
        assert self._mean.shape == array_like.shape, 'Shape mismatch: {} vs {}'.format(
            self._mean.shape, array_like.shape
        )
        self._processed += 1
        self._mean += (array_like - self._mean) / self._processed

    def aggregation(self):
        if self._normalize:
            return preprocessing.normalize(self._mean, norm = 'l1')
        return self._mean

class TrivialAggregator():
    """
    Just keep the first seen array. This is useful to maintain the interface
    when 'aggregating' singletons.
    """
    def __init__(self, representative):
        self._representative = representative

    def consume(self, *args, **kwargs):
        pass

    def aggregation(self):
        return self._representative

_AVAILABLE_AGGREGATORS = { 'count' : CountAggregator,
                           'mean' : MeanAggregator,
                           'trivial' : TrivialAggregator, }

def curried_aggregate_predict_n_by(methods):
    """
    Curried variant of `aggregate_predict_n_by`.
    """
    def _aggregate_predict_n_by(predict_n_result):
        return aggregate_predict_n_by(methods, predict_n_result)
    return _aggregate_predict_n_by
