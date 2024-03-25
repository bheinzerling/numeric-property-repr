from abc import ABC

import numpy as np

from stringcase import snakecase

from util import SubclassRegistry


def diff_signs(values):
    return np.sign(values.diff()[1:].values)


class EditMetric(SubclassRegistry):
    def __call__(self, weights, outputs):
        raise NotImplementedError

    @classmethod
    def get_subclass_names(cls, snakecase=False):
        subclass_names = super().get_subclass_names(snakecase=snakecase)
        subclass_names.remove('scipy_stats_metric')
        return subclass_names


class SignConsistency(EditMetric):
    def __call__(self, *args, **kwargs):
        return self._calculate(*args, **kwargs)

    def _calculate(self, weights, outputs, strict=False, tolerance=0):

        diff_w = diff_signs(weights)
        diff_o = diff_signs(outputs)

        same = diff_w == diff_o

        zero_mask = np.abs(diff_o) <= tolerance
        n_zero_matches = zero_mask.sum()
        n_non_zero_matches = same[~zero_mask].sum()
        n_diffs = len(same)
        n_possible_non_zero_matches = n_diffs - n_zero_matches
        # maybe flip sign
        n_non_zero_matches = max(
            n_non_zero_matches,
            n_possible_non_zero_matches - n_non_zero_matches)

        n_matches = n_non_zero_matches
        if not strict:
            n_matches += n_zero_matches

        score = n_matches / n_diffs
        return score


class SignConsistencyStrict(SignConsistency):
    def __call__(self, weights, outputs, tolerance=0):
        return self._calculate(
            weights, outputs, strict=True, tolerance=tolerance)


class Monotonicity(EditMetric):
    def __call__(self, weights, outputs, tolerance=0):
        diff_o = diff_signs(outputs)
        diff_o = diff_o[np.absolute(diff_o) > tolerance]
        return np.absolute(diff_o.mean())


class MonotonicityStrict(EditMetric):
    def __call__(self, weights, outputs):
        diff_o = diff_signs(outputs)
        return np.absolute(diff_o.mean())


class ScipyStatsMetric(EditMetric, ABC):
    def __call__(self, weights, outputs):
        import scipy.stats
        metric_fn = getattr(scipy.stats, self.fn_name)
        return metric_fn(weights, outputs).statistic


class SpearmanCorrelation(ScipyStatsMetric):
    fn_name = 'spearmanr'


class PearsonCorrelation(ScipyStatsMetric):
    fn_name = 'pearsonr'


class KendallTau(ScipyStatsMetric):
    fn_name = 'kendalltau'


def calculate_edit_metrics(weights, outputs):
    classes = set(EditMetric.get_subclasses())
    classes.remove(ScipyStatsMetric)

    def calculate_metric(metric_cls):
        metric = metric_cls()
        name = snakecase(metric.__class__.__name__)
        score = metric(weights, outputs)
        return name, score

    return dict(map(calculate_metric, classes))
