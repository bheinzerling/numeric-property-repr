from pathlib import Path
import re
from types import SimpleNamespace
from functools import cache

import torch

import numpy as np

from util import (
    cached_property,
    concat_dict_values,
    jsonlines_load,
    WithLog,
    )

from text_encoder_mixin import TransformerRepr
from data_util import WithSplits
from data_numeric_verbalizer import WithVerbalizer


class NumPropData(
        WithLog,
        WithSplits,
        WithVerbalizer,
        TransformerRepr,
        ):
    args = [
        ('--numprop-train-file', dict(type=Path)),
        ('--numprop-dev-file', dict(type=Path)),
        ('--numprop-test-file', dict(type=Path)),
        ('--numprop-train-size', dict(type=int)),
        ('--numprop-dev-size', dict(type=int)),
        ('--numprop-test-size', dict(type=int)),
        ('--numprop-train-file-neg', dict(type=Path)),
        ('--numprop-probe-target', dict(
            type=str,
            choices=[
                'value',
                'value_norm',
                'value_standard',
                'value_quantile',
                'value_quantile_normal',
                'value_power',
                ],
            default='value',
        )),
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        WithLog.__init__(self)
        TransformerRepr.__init__(self, *args, **kwargs)

    @property
    def conf_fields(self):
        fields = [
                'numprop_train_file',
                'numprop_dev_file',
                'numprop_test_file',
                'numprop_train_size',
                'numprop_dev_size',
                'numprop_test_size',
                'numprop_train_file_neg',
                'numprop_probe_target',
            ]
        return fields

    @property
    def train_file(self):
        return self.conf.numprop_train_file

    @property
    def dev_file(self):
        return self.conf.numprop_dev_file or self.split_file_for('dev')

    @property
    def test_file(self):
        return self.conf.numprop_test_file or self.split_file_for('test')

    def split_file_for(self, split_name):
        if 'global_dev' in self.train_file.name:
            return self.train_file
        split_file = self.infer_split_file_from_train_file(
            self.train_file, split_name)
        self.log(f'Inferred {split_name} file: {split_file}')
        return split_file

    @staticmethod
    def infer_split_file_from_train_file(train_file, split_name):
        split_fname = train_file.name.replace('train', split_name)
        return train_file.parent / split_fname

    @staticmethod
    def maybe_add_split_files(conf):
        for split_name in ('dev', 'test'):
            split_attr = f'numprop_{split_name}_file'
            split_file = getattr(conf, split_attr, None)
            if split_file is None:
                split_file = NumPropData.infer_split_file_from_train_file(
                    conf.numprop_train_file, split_name)
                setattr(conf, split_attr, split_file)

    def load_raw_split(self, split_name):
        split_file = getattr(self, f'{split_name}_file')
        n_inst = getattr(self.conf, f'numprop_{split_name}_size')
        instances = list(jsonlines_load(split_file, max=n_inst))
        if n_inst is not None:
            err_msg = f'{split_file}\n{len(instances)} != {n_inst}'
            assert len(instances) == n_inst, err_msg
        return instances

    @staticmethod
    def prop_id_from_sample_file(sample_file):
        fname = Path(sample_file).name
        # ?: specifies a non-capturing group
        match = re.search('(P[0-9]+(?:.lat|.long)?).', fname)
        return match.groups()[0]

    @cached_property
    def prop_id(self):
        return self.prop_id_from_sample_file(self.train_file)

    def instance_id(self, instance):
        return instance['entity_id']

    @cached_property
    def number_type_hint(self):
        return {
            'P569': int,
            'P570': int,
            'P571': int,
            'P1082': int,
            'P2031': int,
            }.get(self.prop_id, float)

    @property
    def numprop_id2unit(self):
        year = ('year', 'years')
        degree = ('degree', 'degrees')
        return {
            'P569': year,
            'P570': year,
            'P571': year,
            'P625.lat': degree,
            'P625.long': degree,
            'P1082': ('people', 'people'),
            'P2031': year,
            'P2044': ('meter', 'meters'),
            'P2046': ('square km', 'square km'),
            }

    @cached_property
    def texts(self):
        return {
            split_name: self._texts(instances)
            for split_name, instances in self.raw.items()
            }

    @cached_property
    def value_scaler(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = concat_dict_values(self.train_raw)
        values = np.array(data['value']).reshape(-1, 1)
        scaler.fit(values)
        return scaler

    @cached_property
    def value_normalizer(self):
        from sklearn.preprocessing import MinMaxScaler
        normalizer = MinMaxScaler()
        instances = self.train_raw + self.dev_raw + self.test_raw
        data = concat_dict_values(instances)
        values = np.array(data['value']).reshape(-1, 1)
        normalizer.fit(values)
        return normalizer

    @cached_property
    def value_quantile_transformer(self):
        from sklearn.preprocessing import QuantileTransformer
        value_transformer = QuantileTransformer(n_quantiles=10000)
        data = concat_dict_values(self.train_raw)
        values = np.array(data['value']).reshape(-1, 1)
        value_transformer.fit(values)
        return value_transformer

    @cached_property
    def value_quantile_normal_transformer(self):
        from sklearn.preprocessing import QuantileTransformer
        value_transformer = QuantileTransformer(
            n_quantiles=10000, output_distribution='normal')
        data = concat_dict_values(self.train_raw)
        values = np.array(data['value']).reshape(-1, 1)
        value_transformer.fit(values)
        return value_transformer

    @cached_property
    def value_power_transformer(self):
        from sklearn.preprocessing import PowerTransformer
        value_transformer = PowerTransformer()
        instances = self.train_raw + self.dev_raw + self.test_raw
        data = concat_dict_values(instances)
        values = np.array(data['value']).reshape(-1, 1)
        value_transformer.fit(values)
        return value_transformer

    def transform(self, value, *, kind=None):
        if kind == 'value':
            return value
        return self._transform(value, kind=kind, direction='transform')

    def inverse_transform(self, value, *, kind=None):
        return self._transform(value, kind=kind, direction='inverse_transform')

    def _transform(self, value, *, kind, direction):
        if kind is None:
            kind = self.conf.numprop_probe_target
        if kind == 'value':
            return value
        transformer = getattr(self, f'{kind}_transformer')
        transform = getattr(transformer, direction)
        is_scalar = not hasattr(value, '__iter__')
        try:
            value_trf = transform(value)
        except ValueError:
            value = np.array(value)
            if value.ndim < 2:
                value = value.reshape(1, -1)
            value_trf = transform(value)
        if is_scalar:
            value_trf = value_trf.item()
        return value_trf

    def tensorize(self, instances, target_only=False, device=None):
        if not instances:
            return {}
        data = concat_dict_values(instances)
        value = np.array(data['value']).reshape(-1, 1)
        data[self.target_tensor_name] = self.transform(value)

        target = torch.tensor(data[self.target_tensor_name]).float()
        if target_only:
            tensors = {
                self.target_tensor_name: target,
                }
        else:
            texts = self._texts(instances)
            texts_enc = self._texts_enc(texts)
            mention_repr = self.pool(texts_enc, instances=instances)
            tensors = {
                **texts_enc,
                'mention_repr': mention_repr,
                self.target_tensor_name: target,
                }
        if 'value' not in tensors:
            tensors['value'] = torch.tensor(data['value'])
        if device is not None:
            tensors = {k: v.to(device=device) for k, v in tensors.items()}
        return tensors

    @cached_property
    def tensors(self):
        return {
            split_name: self.tensorize(instances)
            for split_name, instances in self.raw.items()
            }

    @cached_property
    def target_tensors(self):
        return {
            split_name: self.tensorize(instances, target_only=True)
            for split_name, instances in self.raw.items()
            }

    @cached_property
    def splits(self):
        from data_util import FixedSplits, TensorDictDataset
        return FixedSplits(**{
            split_name: TensorDictDataset(**tensors)
            for split_name, tensors in self.tensors.items()
        })

    @property
    def target_tensor_name(self):
        return self.conf.numprop_probe_target

    @cached_property
    def max_pred(self):
        if self.target_tensor_name == 'value_power':
            return self.target_tensors['train'][self.target_tensor_name].max()
        return None

    @cached_property
    def min_pred(self):
        if self.target_tensor_name == 'value_power':
            return self.target_tensors['train'][self.target_tensor_name].min()
        return None

    def get_Xy(self, split_name):
        y_key = self.conf.numprop_probe_target
        X = self.tensors[split_name]['mention_repr']
        y = self.tensors[split_name][y_key].cpu().numpy()
        self.log(f'tensor sizes {split_name}: X: {X.shape} y: {y.shape}')
        return X, y

    @cache
    def token_mention_start_end_idxs(self, split_name):
        enc_out = self.tensors[split_name]
        instances = self.raw[split_name]
        token_idxs = self.selection_idx(enc_out, instances=instances)
        start_idxs = self.selection_idx_mention_first(enc_out, instances)
        end_idxs = self.selection_idx_mention_last(enc_out, instances)
        return SimpleNamespace(
            token_idxs=token_idxs,
            mention_start_idxs=start_idxs,
            mention_end_idxs=end_idxs,
            )
