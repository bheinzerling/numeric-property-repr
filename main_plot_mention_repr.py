from itertools import product

import numpy as np

from scipy.stats import spearmanr
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.interpolate import griddata

from util import (
    concat_dict_values,
    plot_embeddings_bokeh,
    Figure,
    simple_imshow,
    )
from dim_reduction import PLSRegression


prop_id2label = {
    'P569': 'Birthyear',
    'P570': 'Death year',
    'P625.lat': 'Latitude',
    'P625.long': 'Longitude',
    'P1082': 'Population',
    'P2044': 'Evelation',
    }

prop_id2num_fmt = {
    'P569': int,
    'P570': int,
    'P1082': int,
    'P2044': int,
    'P625.long': '{:.2f}'.format,
    'P625.lat':  '{:.2f}'.format,
    }

cmaps = [
    'rocket',
    'viridis',
    'flare',
    'crest',
    'Reds',
    'Greens',
    'Blues',
    'Greys',
    'Oranges',
    'Purples',
    ]


class MainPlotMentionRepr:
    args = [
        ('--plot-mention-repr-marker-size', dict(type=float, default=2)),
        ('--plot-mention-repr-marker-alpha', dict(type=float, default=0.5)),
        ('--plot-mention-repr-clip-color', dict(action='store_true')),
        ]

    def plot_mention_repr(self):
        X_train, y_train = self.data.get_Xy('train')
        X_test, y_test = self.data.get_Xy('dev')

        model = PLSRegression(n_components=2)
        model.fit(X_train, y_train)
        eval_result = model.eval(X_test, y_test)
        print(eval_result)

        for split_name in 'train', 'test':
            emb = self.data.tensors[split_name]['mention_repr']
            emb2d = model.transform(emb)
            self._plot_mention_repr_for_split(emb2d, split_name)

    def _plot_mention_repr_for_split(self, emb2d, split_name):
        clip_color = self.conf.plot_mention_repr_clip_color
        data_str = self.conf.numprop_train_file.stem
        n_inst = len(emb2d)
        tooltip_fields = concat_dict_values(self.data.raw[split_name])
        texts = self.data.texts[split_name]
        tooltip_fields['text'] = texts
        color = tooltip_fields['value']

        if clip_color:
            color = list(np.array(color).clip(1800, 2000))

        verbalizer = self.conf.numprop_verbalizer
        layer = self.data.text_enc_layer_absolute

        fname_prefix_parts = [
            'mention_repr',
            data_str,
            split_name,
            f'n_inst_{n_inst}',
            f'verbalizer_{verbalizer}',
            self.data.transformer_conf_str,
            f'layer_{layer}',
            ]

        if clip_color:
            fname_prefix_parts.append('clip_color')

        df = pd.DataFrame()
        x_col = 'PLS component 1'
        y_col = 'PLS component 2'
        df[x_col] = emb2d[:, 0]
        df[y_col] = emb2d[:, 1]

        # flip sign of component laodings if necessary
        for col in x_col, y_col:
            corr = spearmanr(df[col], tooltip_fields['value'])
            df[col] *= np.sign(corr.statistic)

        title = '.'.join(fname_prefix_parts)

        self._plot_mention_repr_bokeh(
            emb2d,
            color=color,
            texts=texts,
            tooltip_fields=tooltip_fields,
            title=title,
            )

        self._plot_mention_repr_static(
            df, x_col=x_col, y_col=y_col, color=color, title=title)

    def _plot_mention_repr_bokeh(
            self,
            emb2d,
            *,
            color,
            texts,
            tooltip_fields,
            title,
            ):
        """Creates a couple different versions of interactive scatter plots
        with bokeh.
        """

        def transform_color(color, cmap_eq_hist):
            if cmap_eq_hist:
                return color
            transform = self.data.value_power_transformer.transform
            return transform(np.array(color).reshape(-1, 1))

        configs = product([True, False], repeat=2)
        for cmap_eq_hist, scatter_labels in configs:
            conf_parts = [
                f'cmap_eq_hist_{cmap_eq_hist}',
                f'scatter_labels_{scatter_labels}',
                ]
            conf_str = '.'.join(conf_parts)
            fname = title + '.' + conf_str + '.html'
            outfile = self.conf.outdir / 'fig' / fname

            _color = transform_color(color, cmap_eq_hist)
            plot_kwargs = dict(
                emb=emb2d,
                labels=texts,
                tooltip_fields=tooltip_fields,
                color=_color,
                colorbar=True,
                outfile=outfile,
                cmap='fire',
                cmap_eq_hist=cmap_eq_hist,
                scatter_labels=scatter_labels,
                marker=(None if scatter_labels else 'circle'),
                )

            if not scatter_labels:
                plot_kwargs['size'] = self.conf.plot_mention_repr_marker_size
                plot_kwargs['alpha'] = self.conf.plot_mention_repr_marker_alpha

            plot_embeddings_bokeh(**plot_kwargs)
            print(outfile)

    def _plot_mention_repr_static(self, df, *, x_col, y_col, color, title):
        # make font a bit larger so it is easier to read in the paper
        sns.set(style='ticks', font_scale=1.43)

        # put data into evenly-sized bins / quantiles so that we can evenly
        # map them onto the colorbar range later
        df['value'] = color
        n_bins = 100
        qc = pd.qcut(df['value'], q=n_bins, precision=1, duplicates='drop')
        n_bins = qc.nunique()

        def bin_norm(v):
            return np.searchsorted(qc.cat.categories.right, v) / (n_bins)

        class HistogramNormalize(Normalize):
            def __call__(self, value, clip=None):
                return bin_norm(value)

        val_min = df['value'].min()
        val_max = df['value'].max()
        # we'll use this normalizer to map values to hues (colors) later
        norm = HistogramNormalize(vmin=val_min, vmax=val_max, clip=True)

        # number format for the colorbar ticks
        fmt = prop_id2num_fmt[self.data.prop_id]

        # prepare grid data for plotting interpolations
        steps = 4000j
        grid_x, grid_y = np.mgrid[
            df[x_col].min():df[x_col].max():steps,
            df[y_col].min():df[x_col].max():steps]
        method2grid = {
            method: griddata(
                points=df[[x_col, y_col]].values,
                values=df['value'],
                xi=(grid_x, grid_y),
                method=method,
                )
            for method in ('linear', 'nearest', 'cubic')
        }

        for cmap in cmaps:
            _title = title + '.' + cmap
            cbar_title = prop_id2label[self.data.prop_id]

            with Figure(_title, figwidth=6, figheight=5):
                ax = sns.scatterplot(
                    data=df,
                    x=x_col,
                    y=y_col,
                    hue='value',
                    palette=cmap,
                    s=3,
                    alpha=0.9,
                    edgecolor=None,
                    hue_norm=norm,
                    )
                plt.xlim(df[x_col].min(), df[x_col].max())
                plt.ylim(df[y_col].min(), df[y_col].max())

                mappable = cm.ScalarMappable(cmap=cmap)
                cbar = ax.figure.colorbar(ax=plt.gca(), mappable=mappable)
                cbar.ax.set_ylabel(cbar_title, labelpad=3, rotation=90)

                idx = list(map(round, cbar.get_ticks() * n_bins))[1:-1]
                mid_labels = list(qc.cat.categories.right[idx])

                new_labels = [val_min] + mid_labels + [val_max]
                new_labels = list(map(fmt, new_labels))

                cbar.set_ticklabels(new_labels)
                ax.get_legend().remove()

            # plot some interpolations
            # these are not very meaningful for analysis but look pretty
            for method, grid in method2grid.items():
                _title = title + '.' + cmap + f'.interpolated_{method}'
                for ftype in ('png', 'pdf'):
                    outfile = self.conf.outdir / 'fig' / f'{_title}.{ftype}'
                    simple_imshow(
                        grid.T,
                        cmap=cmap,
                        figsize=(9, 7),
                        origin='lower',
                        outfile=outfile,
                        xlabel=x_col,
                        ylabel=y_col,
                        xticks=False,
                        yticks=False,
                        cbar_title=cbar_title,
                        )
                    print(outfile)
