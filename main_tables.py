from pathlib import Path
from functools import reduce

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib as mpl

from util import (
    df_to_latex,
    mkdir,
    )


class MainTables:
    args = [
        ('--example-inputs-outputs-file', dict(type=Path)),
        ('--example-table-supertabular', dict(action='store_true')),
        ('--example-table-caption', dict(type=str)),
        ('--example-table-raw-output', dict(action='store_true')),
        ]

    def write_dim_reduction_result_table(self):
        """This method reads the data produced by multiple dim_reduction()
        runs and produces a table showing the number of components required
        for a given goodness of fit, as the one in Appendix C, Table 4 of
        our paper.
        """
        results_dir = self.conf.outdir / 'dim_reduction'
        dfs = list(tqdm(map(
            lambda f: pd.read_json(f, orient='table'),
            results_dir.iterdir())))
        df = pd.concat(dfs, ignore_index=True)
        del df['index']

        trf2model = {
            'meta-llama/Llama-2-13b-hf': 'Llama 2 13B',
            'meta-llama/Llama-2-7b-hf': 'Llama 2 7B',
            'mistralai/Mistral-7B-Instruct-v0.1': 'Mistral 7B',
            'tiiuae/falcon-7b-instruct': 'Falcon 7B',
        }

        df['Model'] = df.model.apply(trf2model.__getitem__)

        df = df[df['Method'] == 'PLS']
        df.reset_index(drop=True, inplace=True)

        score_col = 'Goodness of fit ($R^2$)'
        levels = (0.95, 0.90, 0.80, 0.7, 0.6, 0.5)
        group_cols = ['prop_id', 'Model']
        comp_col = '#components'

        max_score_df = df.iloc[df.groupby(group_cols)[score_col].idxmax()]
        _min = max_score_df.groupby(group_cols)[comp_col].min()
        _max = max_score_df.groupby(group_cols)[comp_col].max()
        assert (_min == _max).all()

        level_df = max_score_df.groupby(group_cols)[comp_col].min()
        level_df = level_df.reset_index()
        level_df = level_df.rename(
            columns={comp_col: '$C\\left[max R^2\\right]$'})
        score_df_grouped = max_score_df.groupby(group_cols)[score_col]
        score_df = score_df_grouped.max().reset_index()
        score_col_short = '$R^2$'
        level_df.insert(2, score_col_short, score_df[score_col])
        level_df[score_col_short] = level_df[score_col_short].apply(
            '{:.2f}'.format)

        level_dfs = [level_df]

        for level in levels:
            level_col = f'$C\\left[\\ge {level:.2f} R^2\\right]$'
            df[level_col] = df[score_col] >= level * df.best_score
            level_df = df[df[level_col]]
            min_n_comp_df = level_df.groupby(group_cols)[comp_col].min()
            min_n_comp_df = min_n_comp_df.reset_index()
            min_n_comp_df = min_n_comp_df.rename(columns={comp_col: level_col})
            level_dfs.append(min_n_comp_df)

        _df = reduce(
            lambda df1, df2: pd.merge(df1, df2, on=group_cols), level_dfs)

        pred_id2label = {
            'P569': 'birthyear',
            'P570': 'death year',
            'P1082': 'population',
            'P2044': 'evelation',
            'P625.long': 'longitude',
            'P625.lat': 'latitude',
            }

        from natsort import index_natsorted
        _df = _df.sort_values(
            by='prop_id',
            key=lambda x: np.argsort(index_natsorted(_df["prop_id"])))
        _df['prop_id'] = _df['prop_id'].apply(
            lambda p: pred_id2label[p] + f' ({p})')
        _df = _df.rename(columns={'prop_id': 'Property'})

        header = ' & '.join(list(_df.columns)).replace('#', '\\#')
        df_latex = df_to_latex(_df, header=header)

        outdir = self.conf.outdir / 'tbl'
        fname = 'dim_reduction_n_comp.tex'
        outfile = outdir / fname
        with outfile.open('w') as out:
            out.write(df_latex)
        self.log(outfile)

    def write_example_table(self):
        inputs_outputs = json_load(self.conf.example_inputs_outputs_file)
        self._write_example_table(inputs_outputs)

    def _write_example_table(self, inputs_outputs, cmap_name='coolwarm'):
        df = pd.DataFrame(inputs_outputs)
        df['output_delta_z_score'] = (df.output - df.value) / df.value_std

        cmap = mpl.colormaps[cmap_name]

        def to_color(v):
            if v == 0:
                return (0, 0, 0, 0)
            return cmap((v / 2) + 0.5)

        def to_colored_latex(value, color, value_format=None):
            rgb = color[:3]
            rgb_str = ', '.join((f'{c:.2f}' for c in rgb))
            if value_format:
                value = value_format.format(value)
            return r'\textcolor[rgb]{' + rgb_str + '}{' + f'{value}' + '}'

        color_scale_factor = 2
        df['output_color'] = (
            color_scale_factor * df['output_delta_z_score']).apply(to_color)

        outdir = mkdir(self.conf.outdir / 'tbl')
        prefix = '.'.join([
            'edit_effect_examples',
            self.conf.exp_name,
            self.data.transformer_conf_str,
            f'prop_{self.data.prop_id}',
            f'inst_{self.conf.edit_inst_split_name}_{self.conf.edit_inst_idx}',
            f'n_steps_{self.conf.n_edit_steps}'
            ])

        dir_dfs = []
        from scipy.stats import spearmanr
        corrs = []

        for dir_idx, dir_df in df.groupby('dir_idx'):
            assert dir_df.weight.max() > 0
            assert dir_df.weight.min() < 0
            min_w = dir_df.weight.min()
            max_w = dir_df.weight.max()
            dir_df['weight_norm'] = dir_df.weight.apply(
                lambda w: -w / min_w if w < 0 else w / max_w).round(2)
            assert dir_df.weight_norm.max() == 1.0
            assert dir_df.weight_norm.min() == -1.0
            assert (0 == dir_df.weight_norm).any()

            half_point = len(dir_df) // 2
            first_mean = dir_df.output[:half_point].mean()
            second_mean = dir_df.output[half_point + 1:].mean()
            if second_mean < first_mean:
                dir_df.weight_idx = list(dir_df.weight_idx.iloc[::-1])

            dir_df['weight_color'] = dir_df.weight_norm.apply(to_color)
            weight_col = '$\\alpha_s$'
            output_col = 'Output'
            if self.conf.example_table_raw_output:
                dir_df[output_col] = dir_df['output_raw']
            else:
                dir_df[output_col] = dir_df['output_raw'].apply(
                    lambda s: s.split('\\n')[0].rstrip('.'))
            dir_df[output_col] = dir_df.apply(
                lambda r: to_colored_latex(r[output_col], r['output_color']),
                axis=1)
            dir_df[weight_col] = dir_df.apply(
                lambda r: to_colored_latex(
                        r['weight_norm'],
                        r['weight_color'],
                        value_format='{:.2f}'),
                axis=1)
            columns = [weight_col, output_col]
            tbl_str = dir_df.to_latex(
                columns=columns,
                column_format='r' * len(columns),
                escape=False,
                index=False,
                )

            fname = f'{prefix}.dir_idx_{dir_idx}.tex'
            outfile = outdir / fname
            with outfile.open('w') as out:
                out.write(tbl_str)
            self.log(outfile)

            dir_dfs.append(dir_df)
            corr = spearmanr(dir_df['weight_norm'], dir_df['output'])
            corrs.append(corr)

        dir_df = pd.concat(dir_dfs)
        pivot_df = dir_df.pivot(
            index='weight_idx',
            columns=['dir_idx'],
            values=[output_col])
        pivot_df.columns = [
            '$y_{s,' + str(idx + 1) + '}$' for _, idx in pivot_df.columns]
        dir_columns = list(pivot_df.columns)
        edit_weights = np.linspace(-1, 1, num=len(pivot_df))
        edit_weights_str = [
            to_colored_latex(w, to_color(w), value_format='{:.2f}')
            for w in edit_weights]
        pivot_df[weight_col] = edit_weights_str
        pivot_df = pivot_df.sort_values('weight_idx', ascending=False)
        pivot_df = pivot_df[[weight_col] + dir_columns]

        # we don't care about the sign of the correlation
        corr_vals = [round(np.abs(corr.statistic), 2) for corr in corrs]
        corr_row = ['$\\rho(\\alpha_s, y_{s,k})$'] + corr_vals
        corr_df = pd.DataFrame([corr_row], columns=list(pivot_df.columns))
        pivot_df = pd.concat([pivot_df, corr_df], ignore_index=True)

        print(pivot_df)

        midrule_after = set([len(pivot_df) - 2])
        bold_max_cols = list(pivot_df.columns)[1:]
        bold_max_row_idxs = set([len(pivot_df) - 1])
        tbl_str = df_to_latex(
            pivot_df,
            midrule_after=midrule_after,
            colname_escape=False,
            col_aligns='r' * len(pivot_df.columns),
            bold_max_cols=bold_max_cols,
            bold_max_row_idxs=bold_max_row_idxs,
            )

        fname = f'{prefix}.dir_idx_all.tex'
        outfile = outdir / fname
        with outfile.open('w') as out:
            out.write(tbl_str)
        self.log(outfile)
