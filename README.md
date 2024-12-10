# Monotonic Representation of Numeric Properties in Language Models

This repository contains code and data for the paper: https://arxiv.org/abs/2403.10381

Table of contents:
- [Finding property-encoding directions](#finding-property-encoding-directions)
- [Plotting projections of entity representations onto the top two PLS components](#plotting-projections-of-entity-representations-onto-the-top-two-pls-components)
- [Directed activation patching](#directed-activation-patching)
- [Citation](#citation)

# Finding property-encoding directions

In section 2 of the paper, we fit partial least-squares (PLS) regression models in order to predict numeric properties of entities from their LM representations.

To replicate this experiment run:

```bash
python main.py dim_reduction \
    --numprop-train-file data/wikidata/numprop_P569.quantile_9.sample_0.train.jsonl \
    --transformer meta-llama/Llama-2-13b-hf \
    --text-enc-pooling mention_last \
    --text-enc-layer-relative 0.3 \
    --numprop-probe-target value_power \
    --numprop-dev-size 100 \
    --numprop-test-size 1000
```

This will:
1. load entity data for the Wikidata property [P569](https://www.wikidata.org/wiki/Property:P569) which is "year of birth"
2. verbalize the entity data according to a prompt template, the default is " In what year was X born?"
3. encode the prompt with a LM, here Llama-2-13B
4. select the specified hidden state for each instance, in this case the hidden state corresponding to the last mention token in layer 0.3
5. fit various regression models to predict the entity's year of birth from their hidden state
6. run a sweep over the `n_components` ($k$ in the paper) parameter of PCA and PLS models
7. print the values of the best setting
8. make a plot of the sweep showing regression goodness-of-fit as a function of the number of components and save it in `out/fig`

If everyhing worked you should see output like this:
```
...
out/614ca535-9006-47dd-8d84-00808905ac7d/eval_results.md
out/fig/dim_reduction.n_components_vs_r_squared.numprop_P569.trf_Llama-2-13b-hf.mention_last_pooling.verbalizer_question.layer_None.layer_rel_0.3.target_value_power.n_components_50.png {}
out/fig/dim_reduction.n_components_vs_r_squared.numprop_P569.trf_Llama-2-13b-hf.mention_last_pooling.verbalizer_question.layer_None.layer_rel_0.3.target_value_power.n_components_50.pdf {}
Name: 13, dtype: object
2024-03-26 10:30:49| out/dim_reduction/dim_reduction.n_components_vs_r_squared.numprop_P569.trf_Llama-2-13b-hf.mention_last_pooling.verbalizer_question.layer_None.layer_rel_0.3.target_value_power.n_components_50.json
best score:
#components                       7
Goodness of fit ($R^2$)    0.906423
```

And the plot should look like this:

![regression goodness of fit as function of n_components](out/fig/dim_reduction.n_components_vs_r_squared.numprop_P569.trf_Llama-2-13b-hf.mention_last_pooling.verbalizer_question.layer_None.layer_rel_0.3.target_value_power.n_components_50.png)

# Plotting projections of entity representations onto the top two PLS components

In Figure 3 of the paper we show projections of entity representations onto the top two components of PLS regressions, colored by numeric attribute values:

![2-d projections of mention representations](out/fig/mention_repr_plots.png)

These can be produced with a command like the following:
```bash
python main.py plot_mention_repr \
    --transformer meta-llama/Llama-2-13b-hf \
    --numprop-train-file data/wikidata/numprop_P569.quantile_9.sample_0.train.jsonl \
    --text-enc-layer-relative 0.3 \
    --text-enc-pooling mention_last \
    --numprop-probe-target value_power
```
This should result in a several plots being saved to `out/fig`, including static plots with various color maps, [interactive bokeh plots](out/fig/mention_repr.numprop_P569.quantile_9.sample_0.train.test.n_inst_17532.verbalizer_question.Llama-2-13b-hf.mention_last_pooling.layer_12.cmap_eq_hist_True.scatter_labels_False.html), and --just for fun-- interpolations of the PLS subspace:

![voronoi tesselation of 2-d projection of Llama-2-13B activation space, colored by birthyear](out/fig/mention_repr.numprop_P569.quantile_9.sample_0.train.test.n_inst_17532.verbalizer_question.Llama-2-13b-hf.mention_last_pooling.layer_12.clip_color.rocket.interpolated_nearest.png)

# Directed activation patching

To see an example of directed activation patching such as the one in Table 2 of the paper, run the following command:

```bash
python main.py patch_activations_directed \
    --transformer meta-llama/Llama-2-13b-hf \
    --numprop-train-file data/wikidata/numprop_P569.quantile_9.sample_0.train.jsonl \
    --numprop-test-size 1000 \
    --text-enc-layer-relative 0.3 \
    --text-enc-pooling mention_last \
    --edit-prompt-suffix " One word answer only: " \
    --edit-max-new-tokens 6 \
    --edit-locus-layer-left-window-size 2 \
    --edit-locus-layer-right-window-size 2 \
    --edit-locus-token-left-window-size 2 \
    --edit-locus-token-right-window-size 1 \
    --edit-locus-token-mode window \
    --n-edit-steps 21 \
    --edit-inst-split-name train \
    --edit-inst-idx 998
```

This will:
1. fit PLS regression models and select the optimal number of components $k$ on a dev set
2. for each of the $k$ components, prepare a range of directed patches
3. apply each patch to the specified patching window
4. record the raw LM output and parse a quantity, if possible

You should see output like the follwing:
```
2024-03-26 11:52:53| dir_idx: 4
0       26.82    In what year was Karl Popper born? One word answer only:       1880.\n
1       24.14    In what year was Karl Popper born? One word answer only:       1880\n In
2       21.46    In what year was Karl Popper born? One word answer only:       1880\n In
3       18.78    In what year was Karl Popper born? One word answer only:       1880\n In
4       16.09    In what year was Karl Popper born? One word answer only:       1880\n In
5       13.41    In what year was Karl Popper born? One word answer only:       1882\n In
6       10.73    In what year was Karl Popper born? One word answer only:       1882\n In
7       8.05     In what year was Karl Popper born? One word answer only:       1882\n In
8       5.36     In what year was Karl Popper born? One word answer only:       1882\n In
9       2.68     In what year was Karl Popper born? One word answer only:       1882\n In
10      0.00     In what year was Karl Popper born? One word answer only:       1902\n In
11      -2.58    In what year was Karl Popper born? One word answer only:       1902\n In
12      -5.15    In what year was Karl Popper born? One word answer only:       1929\n In
13      -7.73    In what year was Karl Popper born? One word answer only:       1929\n In
14      -10.31   In what year was Karl Popper born? One word answer only:       1954\n In
15      -12.89   In what year was Karl Popper born? One word answer only:       1964\n In
16      -15.46   In what year was Karl Popper born? One word answer only:       1968\n In
17      -18.04   In what year was Karl Popper born? One word answer only:       1968\n In
18      -20.62   In what year was Karl Popper born? One word answer only:       2012\n\n
19      -23.20   In what year was Karl Popper born? One word answer only:       2012\n\n
20      -25.77   In what year was Karl Popper born? One word answer only:       2012\n\n
```
And also a .tex file like `out/tbl/edit_effect_examples.dev.Llama-2-13b-hf.mention_last_pooling.prop_P569.inst_train_998.n_steps_21.dir_idx_all.tex` which, when rendered, shows the parsed values with a colormap applied:

<img src="out/fig/directed_activation_patching_example_outputs.png" width="50%">

# Analyzing effects and side effects

Figure 5 of the paper shows aggregated effects and side effects of directed activation patching. The diagonals in each subfigure correspond to effects and off-diagonal entries to side-effects. The effect values can be obtained by running the `patch_activations_directed` command from the previous section for a large number of entities (e.g., 100) and averaging the results. To check the side effects on a non-targeted property, we first fit a PLS model on a target data, e..g. birth year (P59), and then perform directed activation patching on non-target data, e.g., elevation above sea level (P2044):

```
python main.py patch_activations_check_side_effects \
	--exp-name check_side_effects \
	--trf-enc-batch-size 8 \
	--transformer meta-llama/Llama-2-7b-hf \
	--numprop-train-file data/wikidata/numprop_P569.quantile_9.sample_0.train.jsonl \
	--numprop-train-size 1000 
	--numprop-dev-size 500 \
	--numprop-test-size 1000 \
	--numprop-verbalizer question \
	--text-enc-layer-relative 0.3 \
	--text-enc-pooling mention_last \
	--suffix-idx 0 \
	--add-initial-space True \
	--edit-prompt-suffix " One word answer only: " \
	--edit-max-new-tokens 6 \
	--edit-locus-layer-left-window-size 2 \
	--edit-locus-layer-right-window-size 2 \
	--edit-locus-token-left-window-size 2 \
	--edit-locus-token-right-window-size \
	--edit-locus-token-mode window \
	--n-edit-steps 40 \
	--numprop-train-file-neg data/wikidata/numprop_P2044.quantile_9.sample_0.train.json \
	--edit-inst-idx 998
```

If everyhing worked you should see output like this:
```
2024-12-10 12:21:37|  How high is Espoo? One word answer only:  -> 100.\nThe [parsed: 100]
2024-12-10 12:21:37| tensor sizes train: X: torch.Size([1000, 4096]) y: (1000, 1)
2024-12-10 12:21:49| edit token wwindow: [6, 7, 8, 9]
2024-12-10 12:21:49| edit layer window: [8, 9, 10, 11, 12]
2024-12-10 12:21:49| dir_idx: 0
0       32.17    How high is Espoo? One word answer only:       100%\n\n
1       30.48    How high is Espoo? One word answer only:       100%\n\n
2       28.78    How high is Espoo? One word answer only:       100%\n\n
3       27.09    How high is Espoo? One word answer only:       100%\n\n
4       25.40    How high is Espoo? One word answer only:       100%\n\n
5       23.70    How high is Espoo? One word answer only:       100%\n\n
6       22.01    How high is Espoo? One word answer only:       100%\n\n
7       20.32    How high is Espoo? One word answer only:       100000
8       18.63    How high is Espoo? One word answer only:       100000
9       16.93    How high is Espoo? One word answer only:       100000
10      15.24    How high is Espoo? One word answer only:       100000
11      13.55    How high is Espoo? One word answer only:       100000
12      11.85    How high is Espoo? One word answer only:       100000
13      10.16    How high is Espoo? One word answer only:       100000
14      8.47     How high is Espoo? One word answer only:       100000
15      6.77     How high is Espoo? One word answer only:       100000
16      5.08     How high is Espoo? One word answer only:       100000
17      3.39     How high is Espoo? One word answer only:       100.\nThe
18      1.69     How high is Espoo? One word answer only:       100.\nThe
19      0.00     How high is Espoo? One word answer only:       100.\nThe
20      -0.94    How high is Espoo? One word answer only:       100.\nThe                                                                                                        21      -1.89    How high is Espoo? One word answer only:       100.\nThe
22      -2.83    How high is Espoo? One word answer only:       100.\nThe
23      -3.78    How high is Espoo? One word answer only:       100.\nThe
24      -4.72    How high is Espoo? One word answer only:       1000.\n
25      -5.66    How high is Espoo? One word answer only:       1000.\n
26      -6.61    How high is Espoo? One word answer only:       1000 feet.
27      -7.55    How high is Espoo? One word answer only:       100%.\n\n
28      -8.50    How high is Espoo? One word answer only:       100%.\n\n
29      -9.44    How high is Espoo? One word answer only:       100%.\n\n
30      -10.38   How high is Espoo? One word answer only:       100%.\n\n
31      -11.33   How high is Espoo? One word answer only:       100%.\n\n
32      -12.27   How high is Espoo? One word answer only:       100%.\n\n
33      -13.22   How high is Espoo? One word answer only:       100%.\n\n
34      -14.16   How high is Espoo? One word answer only:       100%.\n\n
35      -15.11   How high is Espoo? One word answer only:       100%.\n\n
36      -16.05   How high is Espoo? One word answer only:       100%.\n\n
37      -16.99   How high is Espoo? One word answer only:       100%.\n\n
38      -17.94   How high is Espoo? One word answer only:       100%.\n everybody
39      -18.88   How high is Espoo? One word answer only:       100%.\n everybody
```

Each cell in a subfigure represents the average correlation between edit strength and model output change, across 100 entities (selected via the ``--edit-inst-idx'' argument) and each subfigure consists of 9x9 cells, which represent the 81 combinations of effects and side-effects across 9 numeric properties.
Assuming you have collect the results of these 100x9x9 runs with ``--exp-name check_side_effects'', the following command will aggregate and plot the results:
```
python main.py analyze_side_effects --exp-name check_side_effects
```

# Citation

```bibtex
@misc{heinzerling2024monotonic,
    title={Monotonic Representation of Numeric Properties in Language Models}, 
    author={Benjamin Heinzerling and Kentaro Inui},
    year={2024},
    eprint={2403.10381},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
