import math
import numpy as np

from scipy.stats import spearmanr

from sklearn.utils.validation import check_is_fitted

from util import cached_property


class DimReductionRegression:
    def __init__(
            self,
            *args,
            n_components,
            ignore_convergence_failure=True,
            **kwargs,
            ):
        super().__init__(*args, **kwargs)
        self.ignore_convergence_failure = ignore_convergence_failure
        self.n_components = n_components

    def fit(self, X, y):
        try:
            self.model.fit(X, y)
        except ValueError:
            if self.ignore_convergence_failure:
                X = np.random.randn(*X.shape)
                self.model.fit(X, y)
            else:
                raise

    def score(self, X, y):
        return self.model.score(X, y)

    def transform(self, X):
        return self.model.transform(X)

    def inverse_transform(self, X):
        return self.model.inverse_transform(X)

    @property
    def method_name(self):
        raise NotImplementedError

    def eval(self, X, y):
        score = self.score(X, y)
        emb = self.transform(X)
        i = self.n_components - 1
        corr = spearmanr(emb[:, i], y)
        return dict(
            n_components=self.n_components,
            method=self.method_name,
            score=score,
            corr=corr.statistic,
            corr_p=corr.pvalue,
            )

    def check_is_fitted(self):
        check_is_fitted(self)

    @property
    def components(self):
        self.check_is_fitted()
        return self._components

    def dim_reduce(self, X, n_target_components, reduce_shape=True):
        assert 1 <= n_target_components <= self.n_components
        X_trf = self.transform(X)
        if reduce_shape:
            X_trf = X_trf[:, :n_target_components]
        else:
            X_trf[:, n_target_components:] = 0
        return X_trf

    def sweep_components(self, X, y, cumulative=True):
        results = []
        for n_target_components in range(1, self.n_components + 1):
            X_red = self.dim_reduce(X, n_target_components, reduce_shape=False)
            if not cumulative and n_target_components > 1:
                X_red[:, :n_target_components] = 0
            X_red = self.inverse_transform(X_red)
            eval_result = self.eval(X_red, y)
            result = dict(
                n_target_components=n_target_components,
                cumulative=cumulative,
                **eval_result,
                )
            results.append(result)
        return results


class PCARegression(DimReductionRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        self.model = make_pipeline(
            StandardScaler(),
            PCA(n_components=self.n_components),
            LinearRegression(),
            )
        self.pca = self.model.named_steps['pca']

    @property
    def method_name(self):
        return 'PCA'

    def transform(self, X):
        return self.pca.transform(X)

    @property
    def _components(self):
        return self.pca.components_


class PLSRegression(DimReductionRegression):
    def __init__(self, *args, n_edit_steps=200, **kwargs):
        super().__init__(*args, **kwargs)
        from sklearn.cross_decomposition import PLSRegression
        self.model = PLSRegression(n_components=self.n_components)
        self.n_edit_steps = n_edit_steps

    @property
    def method_name(self):
        return 'PLS'

    @property
    def _components(self):
        component_mask = np.eye(self.n_components)
        return self.model.inverse_transform(component_mask)

    @cached_property
    def component2min_edit_weight(self):
        # x_scores_ are the projections on each component. We can use
        # the min and max scores as an empirical (i.e., derived from
        # the training data) estimate of the "natural" range of the
        # weights of each componenta. Intuitively, this should allow
        # to more controllable editing than arbitrarily picking a
        # weight range such as [-3, 3].
        return self.model.x_scores_.min(axis=0)

    @cached_property
    def component2max_edit_weight(self):
        return self.model.x_scores_.max(axis=0)

    @cached_property
    def component_direction_mask(self):
        # assumption: the sign of each entry in y_loadings_ corresponds
        # to the "sign" of the direction in which each component
        # correlates with y
        return self.model.y_loadings_ > 0

    @cached_property
    def n_edit_steps_half(self):
        return math.ceil(self.n_edit_steps / 2)

    @cached_property
    def edit_range_normalized(self):
        n_steps_half = self.n_edit_steps_half
        maybe_add_one = int(2 * n_steps_half == self.n_edit_steps)
        neg_one_to_zero_steps = np.linspace(-1, 0, n_steps_half)
        actual_n = n_steps_half + maybe_add_one
        zero_to_one_steps = np.linspace(0, 1, actual_n)[1:]  # exclude zero
        steps = np.concatenate([neg_one_to_zero_steps, zero_to_one_steps])
        assert steps[n_steps_half - 1] == 0
        return steps

    @cached_property
    def edit_range(self):
        # select the start and end of each component's edit range so
        # that stepping through the range should have the effect of
        # increasing y.
        mask = self.component_direction_mask
        min_weights = self.component2min_edit_weight
        max_weights = self.component2max_edit_weight

        edit_range_start = np.where(mask, min_weights, max_weights)
        edit_range_end = np.where(~mask, min_weights, max_weights)

        steps = self.edit_range_normalized

        scale = np.zeros((self.n_edit_steps, self.n_components))
        scale[steps < 0] = -edit_range_start
        scale[steps > 0] = edit_range_end

        scaled_steps = scale * steps.reshape(-1, 1)
        assert (scaled_steps[self.n_edit_steps_half - 1] == 0).all()
        assert (scaled_steps[0] * scaled_steps[-1] <= 0).all()

        return scaled_steps

    def edit_range_lm_repr__scale_component(self, component_idx):
        """Returns edit vectors in the LM's representational space.
        The edit vectors are all linearly dependent and are obtained by
        scaling a PLS component/direction in PLS space and then mapping
        the scaled component into the LM's representational space.
        """
        component_mask = np.zeros_like(self.edit_range)
        component_mask[:, component_idx] = 1
        component_edit_range_pls = self.edit_range * component_mask
        # map the edits from the PLS component space to the LM's
        # representational space
        edit_range_lm_repr = self.model.inverse_transform(
            component_edit_range_pls)

        # we're only interested in the direction the components,
        # so we subtract the intercepts
        zeros = np.zeros_like(self.edit_range)
        lm_repr_offset = self.model.inverse_transform(zeros)
        edit_range_lm_repr -= lm_repr_offset
        assert (edit_range_lm_repr[self.n_edit_steps_half - 1] == 0).all()
        return edit_range_lm_repr

    def edit_range_lm_repr__scale_lm_repr(self, component_idx):
        """Returns edit vectors in the LM's representational space.
        The edit vectors are all linearly dependent and are obtained
        by scaling a PLS component/direction in PLS space and then
        mapping the scaled component into the LM's representational space.
        """
        component_mask = np.zeros_like(self.edit_range)
        component_mask[:, component_idx] = 1
        # map the edits from the PLS component space to the LM's
        # representational space
        lm_repr = self.model.inverse_transform(component_mask)
        # scale repr by edit range weights
        edit_range = self.edit_range_normalized.reshape(-1, 1)
        edit_range_lm_repr = edit_range * lm_repr
        assert (edit_range_lm_repr[self.n_edit_steps_half - 1] == 0).all()
        return edit_range_lm_repr

    def edit_range_lm_repr_all_components(self):
        edit_range_lm_reprs = [
            self.edit_range_lm_repr(component_idx)
            for component_idx in range(self.n_components)
            ]
        return np.stack(edit_range_lm_reprs)

    def component_edit_range_weight(self, component_idx):
        return self.edit_range[:, component_idx]

    @cached_property
    def component_edit_range_idxs(self):
        return np.arange(self.n_edit_steps)
