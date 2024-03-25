from pathlib import Path
import json

from util import (
    WithRandomState,
    EntryPoint,
    WithLog,
    json_dump,
    mkdir,
    ensure_serializable,
    cached_property,
    )

from main_dim_reduction import MainDimReduction
from main_plot_mention_repr import MainPlotMentionRepr
from main_tables import MainTables
from main_patch_activations import MainPatchActivations
from main_util import MainUtil
from data import NumPropData

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import pandas as pd  # NOQA


class Main(
        EntryPoint,
        WithLog,
        WithRandomState,
        MainDimReduction,
        MainPlotMentionRepr,
        MainTables,
        MainPatchActivations,
        MainUtil,
        ):

    args = ([
        ('--outdir', dict(type=Path, default=Path('out'))),
        ('--runid', dict(type=str)),
        ('--exp-name', dict(type=str, default='dev')),
        ] +
        MainDimReduction.args +
        MainPlotMentionRepr.args +
        MainPatchActivations.args +
        MainTables.args
        )

    def interactive(self):
        breakpoint()

    @cached_property
    def data(self):
        return NumPropData(self.conf)

    @property
    def rundir(self):
        return self.conf.outdir / self.conf.runid

    @property
    def inputs_outputs_file(self):
        return self.rundir / 'inputs_outputs.json'

    def save_inputs_outputs(self, inputs_outputs):
        outfile = self.inputs_outputs_file
        json_dump(inputs_outputs, outfile)
        self.log(str(outfile))
        return outfile

    def log_metrics(self, metrics):
        for metric, score in metrics.items():
            metric_file = self.rundir / f'metric.{metric}.jsonl'
            metric_dict = {metric: score}
            with metric_file.open('a') as out:
                out.write(json.dumps(metric_dict) + '\n')

        metrics_file = self.rundir / 'metrics.jsonl'
        with metrics_file.open('a') as out:
            out.write(json.dumps(metrics) + '\n')

    @property
    def results_dir(self):
        return mkdir(self.conf.outdir / 'results' / self.conf.exp_name)

    def save_results(self, exp_params, metrics):
        run_info = {'runid': self.conf.runid}
        results = dict(**exp_params, **run_info, **metrics)
        fname = (self.conf.runid or 'results') + '.json'
        results_file = self.results_dir / fname
        results = ensure_serializable(results)
        json_dump(results, results_file)
        self.log(results_file)

    @property
    def exp_params(self):
        conf_fields = self.all_conf_fields + self.data.all_conf_fields
        return {field: getattr(self.conf, field) for field in conf_fields}


if __name__ == "__main__":
    Main().run()
