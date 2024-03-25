from itertools import product

import numpy as np
import pandas as pd
import torch

import seaborn as sns
import matplotlib.pyplot as plt

from util import (
    Figure,
    mkdir,
    json_dump_pandas,
    simple_imshow,
    make_and_set_rundir,
    )

from transformer_mixin import TransformerEncoder

from dim_reduction import (
    PCARegression,
    PLSRegression,
    )


class MainDimReduction:
    args = [
        ('--dim-reduction-n-components', dict(type=int, default=50)),
        ]

    def fit_regressions(self, *, X_train, y_train, X_test, y_test):
        total_n_components = self.conf.dim_reduction_n_components
        eval_results = []
        for n_components in range(1, total_n_components + 1):
            for model_cls in PCARegression, PLSRegression:
                model = model_cls(n_components=n_components)
                model.fit(X_train, y_train)
                eval_result = model.eval(X_test, y_test)
                eval_results.append(eval_result)
        return pd.DataFrame(eval_results)

    def dim_reduction(self):
        """This method fits a sequence of regressions with k components
        to predict numeric attributes from LM representations.
        The resulting (n_components, goodness-of-fit) pairs are plotted,
        resulting in a figure like the subfigures showin in Figure 2 of
        our paper.
        """
        make_and_set_rundir(self.conf, log=self.log)

        X_train, y_train = self.data.get_Xy('train')
        X_test, y_test = self.data.get_Xy('dev')

        # get regression results for the unaltered setting
        df = self.fit_regressions(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            )

        # get regression results for the shuffled-labels control
        X_train_shuffled = X_train[torch.randperm(len(X_train))]
        X_test_shuffled = X_test[torch.randperm(len(X_test))]
        shuffled_df = self.fit_regressions(
            X_train=X_train_shuffled,
            y_train=y_train,
            X_test=X_test_shuffled,
            y_test=y_test,
            )
        shuf_str = ' (shuffled labels)'
        shuffled_df['method'] = shuffled_df.method.apply(
            lambda s: s + shuf_str)

        # get regression results for the random-representations control
        X_train_random = torch.normal(
            mean=X_train.mean(), std=X_train.std(), size=X_train.size())
        X_test_random = torch.normal(
            mean=X_test.mean(), std=X_test.std(), size=X_test.size())
        random_df = self.fit_regressions(
            X_train=X_train_random,
            y_train=y_train,
            X_test=X_test_random,
            y_test=y_test,
            )
        rnd_str = ' (random reprs.)'
        random_df['method'] = random_df.method.apply(lambda s: s + rnd_str)

        # put all results into a single dataframe
        df = pd.concat([df, shuffled_df, random_df], ignore_index=True)

        # save raw regression results as a table
        _df = df.pivot_table(
            index='n_components',
            columns=['method'],
            values=['score', 'corr', 'corr_p'],
            )
        _df.columns = _df.columns.to_flat_index().str.join('_')
        print(df)
        outfile = self.conf.rundir / 'eval_results.md'
        _df.to_markdown(outfile)
        print(outfile)

        # plot goodness-of-fit as a function of n_components
        prefix = 'dim_reduction.n_components_vs_r_squared'
        c = self.conf
        conf_str = '.'.join([
            f'numprop_{self.data.prop_id}',
            f'trf_{self.data.transformer_conf_str}',
            f'verbalizer_{c.numprop_verbalizer}',
            f'layer_{c.text_enc_layer}',
            f'layer_rel_{c.text_enc_layer_relative}',
            f'target_{c.numprop_probe_target}',
            f'n_components_{c.dim_reduction_n_components}',
            ])
        title = prefix + '.' + conf_str
        comp_col = '#components'
        score_col = 'Goodness of fit ($R^2$)'
        method_col = 'Method'
        df.rename(columns={
            'n_components': comp_col,
            'score': score_col,
            'method': method_col,
            },
            inplace=True,
            )
        order = [
            'PLS',
            'PCA',
            'PLS' + shuf_str,
            'PCA' + shuf_str,
            'PLS' + rnd_str,
            'PCA' + rnd_str,
            ]

        ymin = min(0, df[score_col].min())

        sns.set(style='ticks', font_scale=1.43)

        with Figure(title):
            ax = sns.lineplot(
                data=df,
                x=comp_col,
                y=score_col,
                hue=method_col,
                style=method_col,
                hue_order=order,
                style_order=order,
                )
            plt.ylim(ymin, 1)
            sns.move_legend(ax, "lower right")

        # print regression results for the best setting
        best_score_idx = df[df[method_col] == 'PLS'][score_col].idxmax()
        best_score_row = df.iloc[best_score_idx][[comp_col, score_col]]
        print('best score:')
        print(best_score_row)

        # store all raw data
        df['model'] = self.conf.transformer
        df['best_score'] = best_score_row[score_col]
        df['prop_id'] = self.data.prop_id
        df['conf_str'] = conf_str
        outdir = mkdir(self.conf.outdir / 'dim_reduction')
        fname = f'{title}.json'
        outfile = outdir / fname
        json_dump_pandas(df, outfile, log=self.log)

    def dim_reduction_edit(self):
        assert self.conf.numprop_probe_target == 'value', 'TODO: scaled probe target values'

        X_train, y_train = self.data.get_Xy('train')
        X_test, y_test = self.data.get_Xy('test')

        from sklearn.cross_decomposition import PLSRegression
        from scipy.stats import spearmanr

        total_n_components = self.conf.dim_reduction_n_components
        regression_data = []
        for n_components in range(1, total_n_components + 1):
            pls = PLSRegression(n_components=n_components)
            try:
                pls.fit(X_train, y_train)
            except ValueError:
                X_train = np.random.randn(*X_train.shape)
                pls.fit(X_train, y_train)
            pls_score = pls.score(X_test, y_test)
            pls_emb = pls.transform(X_train)
            i = n_components - 1
            pls_corr = spearmanr(pls_emb[:, i], y_train)

            regression_data.append(dict(
                n_components=n_components,
                method='PLS',
                model=pls,
                score=pls_score,
                corr=pls_corr.statistic,
                corr_p=pls_corr.pvalue,
                ))

        best_run = sorted(regression_data, key=lambda d: d['score'])[-1]

        n_edit_steps = 200
        # x_scores_ are the projections on each component. We can use
        # the min and max scores as an empirical (i.e., derived from
        # the training data) estimate of the "natural" range of the
        # weights of each componenta. Intuitively, this should allow
        # to more controllable editing than arbitrarily picking a
        # weight range such as [-3, 3].
        component2min_edit_weight = pls.x_scores_.min(axis=0)
        component2max_edit_weight = pls.x_scores_.max(axis=0)

        # ASSUMPTION: the sign of each entry in y_loadings_ corresponds
        # to the "sign" of the direction in which each component correlates with y
        direction_mask = pls.y_loadings_ > 0
        # select the start and end of each component's edit range so
        # that stepping through the range should have the effect
        # of increasing y. That is, we edit the representations to increase
        # the numerical attribute
        edit_range_start = np.where(direction_mask, component2min_edit_weight, component2max_edit_weight)
        edit_range_end = np.where(~direction_mask, component2min_edit_weight, component2max_edit_weight)
        steps = np.linspace(0, 1, n_edit_steps)
        scale = edit_range_end - edit_range_start
        scaled_steps = scale * steps.reshape(-1, 1)
        edit_range_pls = scaled_steps + edit_range_start
        assert edit_range_pls.shape == (n_edit_steps, n_components)

        def get_split_data(split_name):
            metric = torch.nn.MSELoss(reduce=False)
            X, y = self.data.get_Xy(split_name)
            preds = pls.predict(X).squeeze(1)
            errors = metric(torch.tensor(preds), torch.tensor(y))
            sort_idx = errors.argsort()
            error_rank = sort_idx.argsort().cpu()
            return dict(X=X, y=y, preds=preds, errors=errors, error_rank=error_rank)

        split_names = ('train', 'test')
        split_data = {
            split_name: get_split_data(split_name)
            for split_name in split_names
            }

        decomposition = 'PLS'
        centered_vals = [False, True]

        for component_idx in range(0, 15):
            # map the edits from the PLS component space to the LM's representational space
            component_mask = np.zeros_like(edit_range_pls)
            component_mask[:, component_idx] = 1
            component_edit_range_pls = edit_range_pls * component_mask
            component_edit_range = pls.inverse_transform(component_edit_range_pls)
            coeff = component_edit_range_pls[:, component_idx]

            for (centered, split_name) in product(centered_vals, split_names):
                X = split_data[split_name]['X']
                error_rank = split_data[split_name]['error_rank']
                idx2error_rank = {i: r.item() for i, r in enumerate(error_rank)}
                edited_emb = X.unsqueeze(1) + component_edit_range[None, :, :]
                n_inst = len(X)
                edited_preds = pls.predict(edited_emb.flatten(0, 1))
                edited_preds = edited_preds.reshape(n_inst, n_edit_steps)

                if centered:
                    preds = split_data[split_name]['preds']
                    edited_preds = edited_preds - preds.reshape(-1, 1)

                inst_idx2edited_preds = dict(enumerate(edited_preds))

                plot_data = {
                    'coeff': coeff,
                    'mean': edited_preds.mean(axis=0),
                    **inst_idx2edited_preds,
                    }
                value_vars = ['mean'] + list(inst_idx2edited_preds.keys())
                plot_data_df = pd.DataFrame(plot_data)
                melt_df = pd.melt(plot_data_df, id_vars=['coeff'], value_vars=value_vars)
                mean_mask = melt_df['variable'] == 'mean'
                mean_plot_df = melt_df[mean_mask]
                plot_df = melt_df[~mean_mask]
                plot_df['error_rank'] = plot_df.variable.apply(lambda v: idx2error_rank[int(v)])

                title = f'edit_direction.exp_name_{self.conf.exp_name}.decomposition_{decomposition}.component_{component_idx}.split_{split_name}.centered_{centered}'
                with Figure(title):
                    sns.lineplot(
                        data=plot_df,
                        x='coeff',
                        y='value',
                        units='variable',
                        linewidth=1,
                        estimator=None,
                        alpha=0.1,
                        hue='error_rank',
                        palette='rocket',
                        )
                    if not mean_plot_df['value'].isna().all():
                        sns.lineplot(
                            data=mean_plot_df,
                            x='coeff',
                            y='value',
                            units='variable',
                            linewidth=2,
                            alpha=1,
                            estimator=None,
                            )

    def dim_reduction_analyze_results(self):
        df = self.exp_logger.results
        new_rows = []
        for row in df.to_dict(orient='records'):
            regression_data = row.pop('regression_data')
            for reg_row in regression_data:
                reg_row.update(row)
                new_rows.append(reg_row)

        # assert len(new_rows) == len(df) * self.conf.dim_reduction_n_components
        df = pd.DataFrame(new_rows)
        df['verbalizer_and_pooling'] = df.numprop_verbalizer + ' | ' + df.text_enc_pooling
        df['prop_id'] = df.numprop_train_file.map(self.data.prop_id_from_sample_file)
        df['numprop_label'] = df.prop_id.map(self.data.prop_id2label)

        df.dropna(axis=1, inplace=True)

        def clean_tokens(tokens):
            tokens = [token.replace('Ġ', '_') for token in tokens]
            # Some tokenizers "need" an iniitial whitespace, some don't. There's probably
            # a less hacky way to deal with this, but for visualization purposes filtering
            # the space symbol ('▁') seems to work
            tokens = [token for token in tokens if token != '▁']
            return tokens

        for transformer in df.transformer.unique():
            tokenizer = TransformerEncoder.load_tokenizer(transformer)
            trf_name = transformer.replace('/', '_')
            transformer_df = df[df.transformer == transformer]
            for prop_id in sorted(df.prop_id.unique()):
                numprop_df = transformer_df[transformer_df.prop_id == prop_id]
                for verbalizer in numprop_df.numprop_verbalizer.unique():
                    verbalizer_df = numprop_df[numprop_df.numprop_verbalizer == verbalizer]
                    verbalizer_cls = self.data.verbalizer_cls(verbalizer)
                    for probe_target in df.numprop_probe_target.unique():
                        target_df = verbalizer_df[verbalizer_df.numprop_probe_target == probe_target]
                        for paraphrase_idx in sorted(verbalizer_df.numprop_paraphrase_idx.unique()):
                            _df = target_df[target_df.numprop_paraphrase_idx == paraphrase_idx]

                            if hasattr(verbalizer_cls, 'templates'):
                                tpl = verbalizer_cls.templates()[prop_id][paraphrase_idx]
                                if _df.add_initial_space.any():
                                    assert _df.add_initial_space.all()
                                    tpl = ' ' + tpl
                                prefix = verbalizer_cls.prefix_of_template(tpl).rstrip()
                                prefix_tokens = clean_tokens(tokenizer.tokenize(prefix))
                                suffix = verbalizer_cls.suffix_of_template(tpl)
                                suffix_tokens = clean_tokens(tokenizer.tokenize(suffix))
                            else:
                                prefix_tokens = []
                                suffix_tokens = []
                            prefix_len = len(prefix_tokens)

                            labels = [
                                '[sentence mean]',
                                *prefix_tokens,
                                '[mention first]',
                                '[mention mean]',
                                '[mention first_last]',
                                '[mention last]',
                                *suffix_tokens,
                                ]

                            text_enc_pooling_map = {
                                'mean_sent': lambda p, s: 0,
                                'prefix_idx': lambda p, s: 1 + p,
                                'mention_first': lambda p, s: prefix_len + 1,
                                'mean_mention': lambda p, s: prefix_len + 2,
                                'mention_first_last': lambda p, s: prefix_len + 3,
                                'mention_last': lambda p, s: prefix_len + 4,
                                'suffix_idx': lambda p, s: prefix_len + 5 + s,
                                'after_mention': lambda p, s: prefix_len + 5 + s,
                                }

                            _df['repr_idx'] = list(map(
                                lambda r: text_enc_pooling_map[r[0]](r[1], r[2]),
                                zip(_df['text_enc_pooling'], _df['prefix_idx'], _df['suffix_idx'])
                                ))

                            idxmax = _df[(_df.method == 'PLS')].groupby(['text_enc_layer_relative', 'repr_idx'])['score'].idxmax()
                            max_df = _df.loc[idxmax]
                            best_score_df = max_df[['text_enc_layer_relative', 'repr_idx', 'score']]
                            plot_df = best_score_df.pivot_table(
                                index='text_enc_layer_relative',
                                columns='repr_idx'
                                ).sort_values('text_enc_layer_relative', ascending=False)
                            plot_df = plot_df.score.clip(lower=0)

                            ylabels = list(map(lambda v: f'{v:.2f}', plot_df.index))

                            def fname_tpl(plot_data):
                                return f'layer_token_heatmap.exp_{self.conf.exp_name}.{plot_data}.numprop_{prop_id}.verbalizer_{verbalizer}.paraphrase_{paraphrase_idx}.probe_target_{probe_target}.transformer_{trf_name}.png'

                            fname = fname_tpl('pls_r2_score')
                            outfile = self.conf.outdir / 'fig' / fname
                            cell_text = plot_df.applymap(lambda v: f'{v:.2f}'.lstrip('0')).values
                            simple_imshow(
                                plot_df,
                                outfile=outfile,
                                xtick_labels=labels,
                                ytick_labels=ylabels,
                                ylabel='Layer (relative)',
                                # xtick_label_rotation=45,
                                colorbar_range=(0, 1),
                                cbar_title='Regression performance (R-squared)',
                                cell_text=cell_text,
                                cell_text_color='white',
                                )
                            self.log(outfile)

                            n_comp_df = max_df[['text_enc_layer_relative', 'repr_idx', 'n_components']]
                            n_comp_plot_df = n_comp_df.pivot_table(
                                index='text_enc_layer_relative',
                                columns='repr_idx'
                                ).sort_values('text_enc_layer_relative', ascending=False)
                            fname = fname_tpl('n_components')
                            outfile = self.conf.outdir / 'fig' / fname
                            simple_imshow(
                                n_comp_plot_df,
                                outfile=outfile,
                                xtick_labels=labels,
                                ytick_labels=ylabels,
                                ylabel='Layer (relative)',
                                # xtick_label_rotation=45,
                                cbar_title='Number of components',
                                )
                            self.log(outfile)

                            fname = fname_tpl('pls_r2_score.text_n_components')
                            outfile = self.conf.outdir / 'fig' / fname
                            simple_imshow(
                                plot_df,
                                figsize=None,
                                outfile=outfile,
                                xtick_labels=labels,
                                ytick_labels=ylabels,
                                ylabel='Layer (relative)',
                                # xtick_label_rotation=45,
                                colorbar_range=(0, 1),
                                cbar_title='Regression performance (R-squared)',
                                cell_text=n_comp_plot_df.values,
                                cell_text_color='white',
                                aspect_equal=False,
                                )
                            self.log(outfile)
