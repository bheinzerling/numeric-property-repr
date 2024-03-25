from collections import defaultdict
from types import SimpleNamespace

import numpy as np

import torch

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from util import (
    cached_property,
    json_load,
    take_singleton,
    clean_filename,
    is_non_string_iterable,
    make_and_set_rundir,
    Figure,
    )

from dim_reduction import PLSRegression
from hooks import (
    LayerForwardHook,
    DirectedEdit,
    )
from edit_metrics import (
    EditMetric,
    calculate_edit_metrics,
    )


class MainPatchActivations:

    args = [
        ('--n-edit-steps', dict(type=int, default=21)),
        ('--edit-inst-idx', dict(type=int)),
        ('--edit-inst-idx-start', dict(type=int, default=995)),
        ('--edit-inst-idx-end', dict(type=int, default=1000)),
        ('--edit-inst-split-name', dict(type=str, default='train')),
        ('--edit-prompt-suffix', dict(type=str, default='')),
        ('--edit-max-new-tokens', dict(type=int)),
        ('--edit-locus-layer-left-window-size', dict(type=int, default=0)),
        ('--edit-locus-layer-right-window-size', dict(type=int, default=0)),
        ('--edit-locus-token-mode', dict(
            type=str,
            default='window',
            choices=['window', 'mention', 'predicate'])),
        ('--edit-locus-token-left-window-size', dict(type=int, default=0)),
        ('--edit-locus-token-right-window-size', dict(type=int, default=0)),
        ('--edit-direction-scale-mode', dict(
            type=str, default='component', choices=['component', 'lm_repr'])),
        ]

    @cached_property
    def pls(self):
        X_train, y_train = self.data.get_Xy('train')
        pls = PLSRegression(
            n_components=self.conf.dim_reduction_n_components,
            n_edit_steps=self.conf.n_edit_steps,
            )
        pls.fit(X_train, y_train)
        return pls

    @cached_property
    def inst_idxs(self):
        """Returns a list of instance indices based on the range specified
        by the command line arguments.
        """
        c = self.conf
        if c.edit_inst_idx is not None:
            idx_start = c.edit_inst_idx
            idx_end = c.edit_inst_idx + 1
        else:
            idx_start = c.edit_inst_idx_start
            idx_end = c.edit_inst_idx_end
        inst_idxs = list(range(idx_start, idx_end))
        self.log(f'instance indices: {inst_idxs}')
        return inst_idxs

    def get_inst(self, split_name, inst_idx, data=None):
        """Return an instance for activation patching. The only
        difference to the original instance is that we add
        a suffix to the original prompt, since we found that this
        generally improves format adherence.
        """
        if data is None:
            data = self.data
        suffix = self.conf.edit_prompt_suffix
        prompt = data.texts[split_name][inst_idx] + suffix
        prompt_len = len(prompt)
        answer = data.raw[split_name][inst_idx]['value']
        return SimpleNamespace(
            split_name=split_name,
            inst_idx=inst_idx,
            prompt=prompt,
            answer=answer,
            prompt_len=prompt_len,
            )

    @cached_property
    def edit_layer_idxs(self):
        """Returns the indices of all layers in the layer-wise edit window."""
        layer = self.data.text_enc_layer_absolute
        min_layer = layer - self.conf.edit_locus_layer_left_window_size
        assert min_layer >= 0
        max_layer = layer + self.conf.edit_locus_layer_right_window_size
        model_n_layers = LayerForwardHook(self.data.trf).n_layers
        assert max_layer < model_n_layers
        layer_idxs = list(range(min_layer, max_layer + 1))
        self.log(f'edit layer window: {layer_idxs}')
        return layer_idxs

    def edit_token_idxs(self, split_name, inst_idx):
        """Returns the indices of all tokens in the token-wise edit window."""
        idxs = self.data.token_mention_start_end_idxs(split_name)

        def min_max_token_idx(conf):
            match conf.edit_locus_token_mode:
                case "window":
                    token_idx = idxs.token_idxs[inst_idx]
                    left_size = self.conf.edit_locus_token_left_window_size
                    right_size = self.conf.edit_locus_token_right_window_size
                    min_token_idx = max(token_idx - left_size, 0)
                    max_token_idx = min(
                        token_idx + right_size, self.conf.max_seq_len)
                case "mention":
                    min_token_idx = idxs.mention_start_idxs[inst_idx]
                    max_token_idx = idxs.mention_end_idxs[inst_idx]
                case _:
                    raise NotImplementedError(conf.edit_locus_token_mode)
            return min_token_idx, max_token_idx

        min_token_idx, max_token_idx = min_max_token_idx(self.conf)
        edit_token_idxs = list(range(min_token_idx, max_token_idx + 1))

        if self.data.tokenizer.padding_side == 'left':
            # the edit token idxs were computed with padding, but the
            # generate() method doesn't use padding, which matters in
            # case of left-padded tokenizers. As a quick fix, substract
            # the number of pad tokens
            subw_len = self.data.subw_len(self.data.texts['train'][inst_idx])
            n_pad_tokens = self.conf.max_seq_len - subw_len
            edit_token_idxs = [idx - n_pad_tokens for idx in edit_token_idxs]

        self.log(f'edit token wwindow: {edit_token_idxs}')
        return edit_token_idxs

    def direction_range_and_weight_range(
            self,
            dir_idx,
            *,
            scale_mode='component',
            dtype,
            device,
            ):
        """Returns a range of vectors along a the PLS component direction
        specified by `dir_idx` and the corresponding weight range.
        """
        pls = self.pls
        repr_func = getattr(pls, f'edit_range_lm_repr__scale_{scale_mode}')
        # shape(dir_range) = (n_edit_steps, n_hidden)
        dir_range = repr_func(dir_idx)
        # shape(weight_range = (n_edit_steps, )
        weight_range = pls.component_edit_range_weight(dir_idx)
        dir_range = torch.tensor(dir_range, dtype=dtype, device=device)
        return dir_range, weight_range

    def patch_activations_on_prompt(
            self, prompt, answer='', edit_token_idxs=None):
        """Thin wrapper around `patch_activations_on_instance`, the only
        thing it does is creating an instance dictionary from the specified
        `prompt`.
        """
        if edit_token_idxs is None:
            edit_token_idxs = [-1]
        inst = SimpleNamespace(
            inst_idx=-1,
            split_name='dummy',
            answer=answer,
            prompt=prompt,
            prompt_len=len(prompt),
            edit_token_idxs=edit_token_idxs,
            )
        return self.patch_activations_on_instance(inst)

    def patch_activations_on_instance(self, inst, data=None):
        """The main method for patching activations. Given an instance
        `inst`, fit a PLS regression, get a range of patches for
        directed activation patching, run inference for all the patches
        and collect LM outputs.
        """
        sample_file = self.data.conf.numprop_train_file
        pred_id_pls = self.data.prop_id_from_sample_file(sample_file)
        inst_edit_data = []
        X_dev, y_dev = self.data.get_Xy('dev')
        dtype = X_dev.dtype
        if data is None:
            data = self.data
        device = data.trf.device

        # store the output produced by the original, unedited model
        unedited_output = self.lm_output(
            inst.prompt, inst.prompt_len, do_log=True)

        # fit a PLS regression and find the best number of components
        # using the dev set
        dev_results = self.pls.sweep_components(X_dev, y_dev)
        pls_df = pd.DataFrame(dev_results)
        n_components = pls_df.iloc[pls_df.score.idxmax()].n_target_components

        # indices of the directions along which we will perform
        # directed activation patching
        dir_idxs = list(range(n_components))
        # edit weight (in the paper we call this "edit step") indices
        weight_idxs = self.pls.component_edit_range_idxs
        # indices of the token positions at which we will apply patches
        if hasattr(inst, 'edit_token_idxs'):
            edit_token_idxs = inst.edit_token_idxs
        else:
            edit_token_idxs = self.edit_token_idxs(
                inst.split_name, inst.inst_idx)
        # indices of the layers in which we will apply patches
        edit_layer_idxs = self.edit_layer_idxs

        # Do directed activation patching for each direction.
        # All the patches are applied iteratively and independently.
        # The patches get removed after each iteration, which means that
        # there is no cumulative effect.
        for dir_idx in dir_idxs:
            self.log(f'dir_idx: {dir_idx}')
            # get the range of patches and weights along the current current
            # direction
            dir_range, weight_range = self.direction_range_and_weight_range(
                dir_idx,
                scale_mode=self.conf.edit_direction_scale_mode,
                dtype=dtype,
                device=device,
                )

            # loop over all patches and weights in the current range
            for _ in zip(weight_idxs, dir_range, weight_range):
                weight_idx, direction, weight = _

                # attach the hook which will perform activation patching
                for layer_idx in edit_layer_idxs:
                    for token_idx in edit_token_idxs:
                        hook = LayerForwardHook(self.data.trf)
                        # DirectedEdit adds a `direction`, i.e.,
                        # a fixed vector offset to hidden states
                        hook_func = DirectedEdit(
                            direction=direction,
                            token_idx=token_idx,
                            call_once=True,
                            )
                        hook.attach(hook_func, layer_idx=layer_idx)

                # run inference with the hooked model and parse a quantity
                # from the output, if possible
                edited_output = self.lm_output(inst.prompt, inst.prompt_len)
                print('\t'.join([
                    f'{weight_idx}',
                    f'{weight:.2f}',
                    inst.prompt,
                    edited_output.raw,
                    ]))
                if inst.split_name == 'dummy':
                    raw_data = {}
                else:
                    raw_data = data.raw[inst.split_name][inst.inst_idx]
                inst_edit_data.append(dict(
                    inst_idx=inst.inst_idx,
                    prompt=inst.prompt,
                    unedited_output_raw=unedited_output.raw,
                    unedited_output_parsed=unedited_output.parsed,
                    true_answer=inst.answer,
                    dir_idx=dir_idx,
                    weight_idx=weight_idx,
                    weight=weight,
                    output_raw=edited_output.raw,
                    output=edited_output.parsed,
                    model=self.conf.transformer,
                    pred_id_pls=pred_id_pls,
                    **raw_data,
                    ))
            print()
        return inst_edit_data

    def calculate_metrics(self, df):
        def to_metrics_df(_df):
            edit_metrics = calculate_edit_metrics(_df['weight'], _df['output'])
            return pd.DataFrame([edit_metrics])

        group_cols = ['inst_idx', 'dir_idx']
        return df.groupby(group_cols).apply(to_metrics_df)

    def save_metrics(self, metrics):
        metrics_json = metrics.to_json(orient='table')
        from pandas.testing import assert_frame_equal
        assert_frame_equal(metrics, pd.read_json(metrics_json, orient='table'))

        metrics_file = self.conf.rundir / self.metrics_fname
        with metrics_file.open('w') as out:
            out.write(metrics_json)
        self.log(metrics_file)

    @staticmethod
    def select_best_metrics(metrics, key_metric='spearman_correlation'):
        metrics_by_direction = metrics.groupby('dir_idx').mean()
        key_metric_df = metrics_by_direction[key_metric].abs()
        best_direction = key_metric_df.argmax()
        # argmax returns -1 if there are only NaN values
        if best_direction == -1:
            # arbitrarily pick direction 0
            best_direction = 0
            idxmax = 0
        else:
            idxmax = key_metric_df.idxmax()
        locator = 'loc' if idxmax == 'mean' else 'iloc'
        best_direction_df = getattr(metrics_by_direction, locator)[idxmax]
        best_direction_metrics = best_direction_df.astype(np.float32).to_dict()
        best_direction_metrics['best_dir_idx'] = int(best_direction)
        return best_direction_metrics

    def save_edit_data(self, edit_data):
        df = pd.DataFrame(edit_data)
        metrics = self.calculate_metrics(df)
        self.save_metrics(metrics)
        inputs_outputs = df.to_dict(orient='records')
        self.save_inputs_outputs(inputs_outputs)
        best_metrics = self.select_best_metrics(metrics)
        self.log_metrics(best_metrics)
        self.save_results(exp_params=self.exp_params, metrics=best_metrics)
        self._write_example_table(inputs_outputs)

    @property
    def metrics_fname(self):
        return 'metrics_all.json'

    def patch_activations_directed(self):
        """Run directed activation patching for the instances specified
        by the command line arguments.
        """
        make_and_set_rundir(self.conf, log=self.log)
        edit_data = []
        for inst_idx in self.inst_idxs:
            inst = self.get_inst(self.conf.edit_inst_split_name, inst_idx)
            inst_edit_data = self.patch_activations_on_instance(inst)
            edit_data.extend(inst_edit_data)
        self.save_edit_data(edit_data)

    def plot_edit_results(self, df, title):
        import seaborn as sns
        weight_col = 'Edit direction weight'
        output_col = 'LM output'
        plot_df = df.rename(columns={
            'weight': weight_col,
            'output': output_col,
            })
        for plot_type in ['scatterplot', 'lineplot']:
            with Figure(f'{title}.{plot_type}'):
                plot_fn = getattr(sns, plot_type)
                plot_fn(
                    data=plot_df,
                    x=weight_col,
                    y=output_col,
                    )

    def patch_activations_check_side_effects(self):
        from copy import deepcopy
        from data import Data

        split_name = 'train'
        assert self.conf.numprop_train_file_neg
        conf = deepcopy(self.conf)
        conf.numprop_train_file = conf.numprop_train_file_neg

        neg_data = Data.get(conf.dataset)(conf)
        neg_inst = self.get_inst(split_name, conf.edit_inst_idx, data=neg_data)
        edit_data = self.patch_activations_on_instance(neg_inst, data=neg_data)
        self.save_edit_data(edit_data)

    @cached_property
    def edit_scores_df(self):
        df = self.exp_logger.results
        metric_cols = EditMetric.get_subclass_names(snakecase=True)
        df[metric_cols] = df[metric_cols].fillna(0)
        return df

    def plot_edit_scope_heatmaps(self):
        from edit_heat_maps import EditHeatMaps
        EditHeatMaps(
            self.conf,
            df=self.edit_scores_df,
            data=self.data,
            log=self.log,
            ).plot_all()

    def load_edit_outputs(self, runid):
        if is_non_string_iterable(runid):
            dfs = map(self._load_edit_outputs, runid)
            return pd.concat(dfs)
        else:
            return self._load_edit_outputs(runid)

    def _load_edit_outputs(self, runid):
        inputs_outputs_file = self.conf.outdir / runid / 'inputs_outputs.json'
        inputs_outputs = json_load(inputs_outputs_file)
        df = pd.DataFrame(inputs_outputs)
        return df

    def analyze_edit_scope_results(self):
        scores_df = self.edit_scores_df
        metrics = EditMetric.get_subclass_names(snakecase=True)
        metric_corrs = scores_df[metrics].abs().corr()
        title = f'exp_{self.conf.exp_name}.metric_corrs'
        with Figure(title):
            sns.heatmap(metric_corrs, annot=True, center=0, cmap='coolwarm_r')
        title += '_abs'
        with Figure(title):
            sns.heatmap(metric_corrs.abs(), annot=True, cmap='Blues')

        print(metric_corrs.abs().mean())
        print('mean correlations:')
        best_metric = metric_corrs.abs().mean().idxmax()
        scores_df[best_metric] = scores_df[best_metric].abs()
        print('best metric:', best_metric)

        hyperparams = [
            'edit_locus_layer_left_window_size',
            'edit_locus_layer_right_window_size',
            'edit_locus_token_left_window_size',
            'edit_locus_token_right_window_size',
            'edit_locus_token_mode',
            'text_enc_layer_relative',
            'text_enc_pooling',
            'prefix_idx',
            'suffix_idx',
            'model',
            ]
        best_setting = defaultdict(lambda: defaultdict(dict))
        for metric in metrics:
            for hyperparam in hyperparams:
                for agg in 'mean', 'max':
                    agg_df = scores_df.groupby(hyperparam).agg(agg)
                    best_setting[agg][hyperparam][metric] = {
                        'setting': agg_df[metric].idxmax(),
                        'score': agg_df[metric].max(),
                        }
        from pprint import pprint
        pprint(best_setting)

    def pred_correct(self, runid):
        inputs_outputs_file = self.conf.outdir / runid / 'inputs_outputs.json'
        inputs_outputs = json_load(inputs_outputs_file)
        unedited_inputs_outputs = [
            io for io in inputs_outputs if io['weight'] == 0]
        true_answer = take_singleton(set(
            io['true_answer'] for io in unedited_inputs_outputs))
        pred = take_singleton(set(
            io['output'] for io in unedited_inputs_outputs))
        unedited_pred = take_singleton(set(
            io['unedited_output_parsed'] for io in unedited_inputs_outputs))
        assert unedited_pred == pred
        return true_answer == pred

    def analyze_side_effects(self):
        results_df = self.exp_logger.results
        results_df['pred_correct'] = results_df.runid.apply(self.pred_correct)
        for correctness in ['any', True, False]:
            title_suffix = f'correctness_{correctness}'
            if correctness == 'any':
                filtered_results_df = results_df
            else:
                filtered_results_df = results_df[results_df.pred_correct == correctness]
            self._analyze_side_effects(
                filtered_results_df,
                title_suffix=title_suffix,
                )

    def _analyze_side_effects(self, results_df, *, title_suffix):
        metrics_df = self.metrics_for_results(results_df)
        metric_col = 'spearman_correlation'
        metrics_df[metric_col] = metrics_df[metric_col].abs()
        pos_numprop_id_col = 'Source property ID'
        neg_numprop_id_col = 'Target property ID'
        pos_col = 'Targeted property'
        neg_col = 'Measured property'
        metrics_df[pos_numprop_id_col] = metrics_df.numprop_train_file.apply(
            self.data.prop_id_from_sample_file)
        metrics_df[neg_numprop_id_col] = metrics_df.numprop_train_file_neg.apply(
            self.data.prop_id_from_sample_file)

        def numprop_id2label(numprop_id):
            default_label = self.data.numprop_id2label(numprop_id)
            short_labels = {
                    'P569': 'birthyear',
                    'P570': 'death year',
                    'P2044': 'elevation',
                    'P2031': 'work period start',
                }
            return short_labels.get(numprop_id, default_label)

        metrics_df[pos_col] = metrics_df[pos_numprop_id_col].apply(numprop_id2label)
        metrics_df[neg_col] = metrics_df[neg_numprop_id_col].apply(numprop_id2label)
        pos_neg_cols = [pos_col, neg_col]
        dir_col = 'dir_idx'
        group_cols = [pos_col, neg_col, dir_col]
        group_df = metrics_df.groupby(group_cols)[metric_col]

        agg_df = group_df.mean().reset_index()

        def find_best_dir_idx(df, pos_val, neg_val, *, mode):
            keep = {
                'max': 'last',
                'min': 'first',
                }[mode]
            comp_df = agg_df[(df[pos_col] == pos_val) & (df[neg_col] == neg_val)]
            dir_row = comp_df.sort_values(metric_col).drop_duplicates(pos_neg_cols, keep=keep)
            dir_idx = dir_row[dir_col].item()
            return dir_idx

        def lookup_pos_metric(row):
            df = agg_df
            mask = (
                (df[pos_col] == row[pos_col]) &
                (df[neg_col] == row[pos_col]) &  # row[pos_col] is intended
                (df[dir_col] == row[dir_col])
                )
            match_df = df[mask]
            if len(match_df) == 1:
                return match_df[metric_col].item()
            return np.nan

        pos_metric_col = metric_col + '_pos'
        neg_metric_col = metric_col + '_neg'
        diff_metric_col = metric_col + '_diff'
        spec_col = 'Specificity'
        side_eff_col = 'Side effect strength'

        agg_df[pos_metric_col] = agg_df.apply(lookup_pos_metric, axis=1)
        agg_df.dropna(subset=[pos_metric_col], inplace=True)
        agg_df[neg_metric_col] = agg_df[metric_col]

        posneg_plot_df = agg_df.sort_values(pos_metric_col).drop_duplicates(pos_neg_cols, keep='last')
        posneg_pivot_df = posneg_plot_df.pivot(pos_col, neg_col, neg_metric_col)
        pos_pivot_df = posneg_plot_df.pivot(pos_col, neg_col, pos_metric_col)
        n = len(posneg_pivot_df)
        for i in range(n):
            posneg_pivot_df.values[i, i] = pos_pivot_df.values[i, i]
        print(posneg_pivot_df)
        title = '.'.join([
            'analyze_side_effects',
            self.conf.exp_name,
            'pos_metric_diag_neg_metric_nondiag',
            title_suffix,
            ])
        with Figure(title):
            cmap = plt.cm.Blues
            sns.heatmap(
                posneg_pivot_df,
                vmin=0,
                vmax=1,
                cmap=cmap,
                cbar_kws={'label': 'Edit effect (Spearman correlation)'},
                annot=True,
                annot_kws={'fontsize': 'small', 'fontweight': 'light'},
                fmt='.2f',
                )

        values = posneg_pivot_df.values
        pos_mean = values.diagonal().mean()
        pos_std = values.diagonal().std()
        neg_mask = ~np.eye(values.shape[0], dtype=bool)
        neg_mean = values[neg_mask].mean()
        neg_std = values[neg_mask].std()
        print(f'mean effect: {pos_mean:.2f}\pm{pos_std:.2f}')
        print(f'mean side effect: {neg_mean:.2f}\pm{neg_std:.2f}')

        agg_df[diff_metric_col] = agg_df[pos_metric_col] - agg_df[metric_col]
        # an edit direction is "specific" if it:
        # - encodes/edits the source/pos numattr, i.e., pos_metric is large
        # - does not encode/edit the target/neg numprop, i.e., neg_metric is small compared to pos_metric
        agg_df[spec_col] = agg_df[pos_metric_col] * (1 - agg_df[neg_metric_col] / agg_df[pos_metric_col]).replace(np.inf, 0)
        plot_df = agg_df.sort_values(spec_col).drop_duplicates(pos_neg_cols, keep='last')
        plot_df[side_eff_col] = 1 / plot_df[spec_col]
        vmax = plot_df[~np.isinf(plot_df[side_eff_col])][side_eff_col].max()

        pivot_df = plot_df.pivot(pos_col, neg_col, side_eff_col)
        pivot_df.replace(np.inf, 0, inplace=True)
        pos_pivot_df = plot_df.pivot(pos_col, neg_col, pos_metric_col)
        assert (pivot_df.index == pos_pivot_df.index).all()
        assert (pivot_df.columns == pos_pivot_df.columns).all()
        print(pivot_df)

        n = len(pivot_df)
        cell_text = [[''] * n for _ in range(n)]
        for i in range(n):
            text_val = pos_pivot_df.values[i, i]
            cell_text[i][i] = '{:.2f}'.format(text_val)

        title = '.'.join([
            'analyze_side_effects',
            self.conf.exp_name,
            'side_effect_strength',
            title_suffix,
            ])
        with Figure(title):
            cmap = plt.cm.Blues
            sns.heatmap(
                pivot_df,
                vmin=0,
                vmax=vmax,
                cmap=cmap,
                cbar_kws={'label': side_eff_col},
                norm=LogNorm(),
                annot=np.array(cell_text),
                fmt='',
                annot_kws={'color': 'black'},
                )

        pivot_df = plot_df.pivot(pos_col, neg_col, spec_col)
        print(pivot_df)
        title = '.'.join([
            'analyze_side_effects',
            self.conf.exp_name,
            'specificity',
            title_suffix,
            ])
        with Figure(title):
            cmap = plt.cm.Blues
            sns.heatmap(
                pivot_df,
                vmin=0,
                vmax=1,
                cmap=cmap,
                cbar_kws={'label': spec_col},
                annot=np.array(cell_text),
                fmt='',
                annot_kws={'color': 'black'},
                )

        max_pos_plot_df = agg_df.sort_values(pos_metric_col).drop_duplicates(pos_neg_cols, keep='last')
        max_pos_metric_pivot_df = max_pos_plot_df.pivot(pos_col, neg_col, pos_metric_col)

        max_pos_plot_df[side_eff_col] = 1 / max_pos_plot_df[spec_col]
        vmax = max_pos_plot_df[~np.isinf(max_pos_plot_df[side_eff_col])][side_eff_col].max()

        max_pos_pivot_df = max_pos_plot_df.pivot(pos_col, neg_col, side_eff_col)
        max_pos_pivot_df.replace(np.inf, 0, inplace=True)
        print(max_pos_pivot_df)

        n = len(max_pos_metric_pivot_df)
        cell_text = [[''] * n for _ in range(n)]
        for i in range(n):
            text_val = max_pos_metric_pivot_df.values[i, i]
            cell_text[i][i] = '{:.2f}'.format(text_val)

        title = '.'.join([
            'analyze_side_effects',
            self.conf.exp_name,
            'max_pos_side_effect_strength',
            title_suffix,
            ])
        with Figure(title):
            cmap = plt.cm.Blues
            sns.heatmap(
                max_pos_pivot_df,
                vmin=0,
                vmax=vmax,
                cmap=cmap,
                cbar_kws={'label': side_eff_col},
                norm=LogNorm(),
                annot=np.array(cell_text),
                fmt='',
                annot_kws={'color': 'black'},
                )

        max_spec_plot_df = agg_df.sort_values(spec_col).drop_duplicates(pos_neg_cols, keep='last')
        max_spec_pivot_df = max_spec_plot_df.pivot(pos_col, neg_col, neg_metric_col)
        max_spec_pos_pivot_df = max_spec_plot_df.pivot(pos_col, neg_col, pos_metric_col)
        n = len(max_spec_pivot_df)
        for i in range(n):
            max_spec_pivot_df.values[i, i] = max_spec_pos_pivot_df.values[i, i]
        print(max_spec_pivot_df)
        title = '.'.join([
            'analyze_side_effects',
            self.conf.exp_name,
            'max_spec_posneg',
            title_suffix,
            ])
        with Figure(title):
            cmap = plt.cm.Blues
            cbar_kws = {'label': 'Edit effect (Spearman correlation $\\rho$)'}
            annot_kws = {'fontsize': 'small', 'fontweight': 'light'}
            sns.heatmap(
                max_spec_pivot_df,
                vmin=0,
                vmax=1,
                cmap=cmap,
                cbar_kws=cbar_kws,
                annot=True,
                annot_kws=annot_kws,
                fmt='.2f',
                )

    def metrics_for_results(self, results_df):

        def metrics_for_runid(run_row):
            runid = run_row.runid
            rundir = self.conf.outdir / runid
            metrics_file = rundir / self.metrics_fname
            metrics_df = pd.read_json(metrics_file, orient='table')
            row_df = run_row.to_frame().T
            duplicate_cols = set(row_df.columns) & set(metrics_df.columns)
            row_df.drop(columns=duplicate_cols, inplace=True)
            row_df = pd.concat([row_df] * len(metrics_df))
            row_df.index = metrics_df.index.to_flat_index()
            concat_df = pd.concat([metrics_df, row_df], axis=1)
            concat_df.index.names = metrics_df.index.names
            return concat_df.reset_index()

        return pd.concat(map(metrics_for_runid, results_df.iloc))

    def plot_edit_outputs(self):
        scores_df = self.edit_scores_df
        scores_df['numprop_id'] = scores_df.numprop_train_file.apply(
            self.data.prop_id_from_sample_file)
        group_cols = ['transformer', 'numprop_id']
        for group_key, group_df in scores_df.groupby(group_cols):
            group_key_vals = dict(zip(group_cols, group_key))
            self._plot_edit_outputs(group_df, **group_key_vals)

    def _plot_edit_outputs(
            self,
            scores_df,
            metric='spearman_correlation',
            select_best=False,
            **group_key_vals,
            ):

        if select_best:
            scores_df[metric] = scores_df[metric].abs()
            idxmax = scores_df[metric].idxmax()
            best_row = scores_df.iloc[idxmax]
            runid = best_row.runid
            df = self.load_edit_outputs(runid)
        else:
            df = self.load_edit_outputs(scores_df.runid)
            runid = 'all'

        scores_df[metric] = scores_df[metric].abs()
        best_dir_idx = scores_df.groupby(
            'best_dir_idx')[metric].mean().idxmax()

        for dir_idx in df.dir_idx.unique():
            title_suffix = '_best' if best_dir_idx == dir_idx else ''
            self.plot_direction_edit_outputs(
                df,
                dir_idx=dir_idx,
                runid=runid,
                metric=metric,
                title_suffix=title_suffix,
                **group_key_vals,
                )

        metrics = EditMetric.get_subclass_names(snakecase=True)
        for metric in metrics:
            scores_df[metric] = scores_df[metric].abs()
            dir_idx = scores_df.groupby(
                'best_dir_idx')[metric].mean().idxmax()
            self.plot_direction_edit_outputs(
                df,
                dir_idx=dir_idx,
                runid=runid,
                metric=metric,
                title_suffix='_best',
                **group_key_vals,
                )

    def plot_direction_edit_outputs(
            self,
            df,
            *,
            dir_idx,
            runid,
            metric,
            title_suffix='',
            outlier_stds=3,
            **group_key_vals,
            ):
        dir_idx_col = 'dir_idx'
        df = pd.DataFrame(df[df[dir_idx_col] == dir_idx])
        if runid != 'all':
            assert len(df) == len(df.inst_idx.unique()) * len(df.weight_idx.unique())
        assert df.weight_idx.min() == 0

        numprop_id = group_key_vals['numprop_id']
        unit, unit_plural = self.data.numprop_id2unit[numprop_id]
        output_delta_col = f'Model output change ({unit_plural})'
        output_col = f'Model output ({unit})'
        weight_col = 'Edit weight (scaled)'
        if numprop_id.startswith('P625.'):
            # reparse since the original parsing method was flawed
            def parse_geocoord(text):
                return self.quantity_parser(text, expect_geocoord=True)
            df['output'] = df.output_raw.apply(parse_geocoord)
            df['unedited_output_parsed'] = df.unedited_output_raw.apply(parse_geocoord)
        df[output_delta_col] = df.output - df.unedited_output_parsed
        df[weight_col] = 2 * (df.weight_idx / df.weight_idx.max()) - 1
        df[output_col] = df.output

        if not df.index.is_unique:
            df.set_index(np.arange(len(df)), inplace=True)

        mean_df = df.groupby(weight_col).mean()
        mean_df[weight_col] = 2 * (mean_df.weight_idx / mean_df.weight_idx.max()) - 1

        is_descending = mean_df[output_delta_col].diff().mean() < 0
        if is_descending:
            df[weight_col] *= -1
            mean_df[weight_col] *= -1

        group_str = '.'.join(f'{k}_{v}' for k, v in group_key_vals.items())
        for y_col in [output_col, output_delta_col]:
            title = '.'.join([
                'dir_edit_outputs',
                self.conf.exp_name,
                group_str,
                y_col,
                f'runid_{runid}'
                f'metric_{metric}',
                f'dir_{dir_idx}',
                ])
            title = clean_filename(title)
            title += title_suffix

            title += '.error_band'
            zoom_title = title + '.zoom'
            zoom_min = -0.25
            zoom_max = 0.5
            zoom_df = df[(zoom_min <= df[weight_col]) & (df[weight_col] <= zoom_max)]
            zoom_mean_df = mean_df[(zoom_min <= mean_df[weight_col]) & (mean_df[weight_col] <= zoom_max)]

            sns.set(style='ticks', font_scale=1.43)

            for _title, _df, _mean_df in [(title, df, mean_df), (zoom_title, zoom_df, zoom_mean_df)]:
                with Figure(_title):
                    sns.lineplot(
                        data=_df,
                        x=weight_col,
                        y=y_col,
                        color='red',
                        )
                    sns.lineplot(
                        data=_mean_df,
                        x=weight_col,
                        y=y_col,
                        color='red',
                        )
