from pathlib import Path
from collections import defaultdict
from functools import cache

from util import (
    cached_property,
    Configurable,
    json_dump,
    json_load,
    )

from templates import _pred2paraphrase_idx2question_tpl
from util import precision_and_type_for_pred_and_unit


class Verbalizer(Configurable):

    def verbalize(self, instance):
        raise NotImplementedError


class EntityVerbalizer(Verbalizer):
    def verbalize(self, instance):
        return instance['entity_label']


class RelationVerbalizer(Verbalizer):
    @cached_property
    def verbalize_subject(self):
        return EntityVerbalizer(self.conf).verbalize

    @cached_property
    def verbalize_object(self):
        raise NotImplementedError

    def template(self, instance):
        raise NotImplementedError

    def template_fillers(self, subj, obj):
        return {'subj': subj, 'obj': obj}

    def verbalize(self, instance):
        subj = self.verbalize_subject(instance)
        obj = self.verbalize_object(instance)
        tpl = self.template(instance)
        return tpl.format(**self.template_fillers(subj, obj))


class QuantityVerbalizer(Verbalizer):
    def verbalize_quantity(self, instance):
        unit_str = self.format_unit(instance)
        joiner = ' ' if unit_str else ''
        return self.format_value(instance) + joiner + unit_str

    def verbalize(self, instance):
        return self.verbalize_quantity(instance)

    def format_value(
            self, instance=None, pred_id=None, unit_id=None, value=None):
        if instance is not None:
            pred_id = instance['pred_id']
            unit_id = instance['unit_id']
            value = instance['value']
        precision, ty = precision_and_type_for_pred_and_unit(pred_id, unit_id)

        if ty is float:
            def format_fn(v):
                return '{0:.{1}f}'.format(v, precision)
        else:
            format_fn = str

        return format_fn(value)

    def value_str_to_number(self, value_str):
        if '.' in value_str:
            return float(value_str)
        return int(value_str)

    @cached_property
    def inflect(self):
        from inflect import engine
        return engine()

    @property
    def singular_units(self):
        return {
            'Q11229',  # percent
            'Q1092296',  # annum
            }

    @property
    def no_unit_units(self):
        return {
            'Q1092296',  # annum
            'Q28390',  # degree
            }

    def format_unit(
            self, instance=None, pred_id=None, unit_id=None, value=None):
        if instance is not None:
            unit_id = instance['unit_id']
            value = instance['value']
        unit_label = instance['unit_label']
        if unit_id in self.no_unit_units:
            return ''
        if unit_label == '1':
            return ''
        if value == 1 or unit_id in self.singular_units:
            return unit_label
        return self.inflect.plural(unit_label)


class MentionAffixes:

    subj_str = '{subj}'

    def prefix(self, template):
        prefix_end_idx = template.index(self.subj_str)
        return template[:prefix_end_idx]

    def prefixes(self, instances):
        prefixes = []
        for inst in instances:
            tpl = self.template(inst)
            prefix = self.prefix(tpl)
            prefixes.append(prefix)
        return prefixes

    def suffix(self, template):
        prefix_end_idx = template.index(self.subj_str)
        suffix_start_idx = prefix_end_idx + len(self.subj_str)
        return template[suffix_start_idx:]

    def suffixes(self, instances):
        suffixes = []
        for inst in instances:
            tpl = self.template(inst)
            suffix = self.suffix(tpl)
            suffixes.append(suffix)
        return suffixes

    def mentions(self, instances):
        return [inst['entity_label'] for inst in instances]


class MentionVerbalizer(EntityVerbalizer):
    def prefixes(self, instances):
        return [''] * len(instances)

    def suffixes(self, instances):
        return [''] * len(instances)

    def mentions(self, instances):
        return [inst['entity_label'] for inst in instances]


class NumericRelationVerbalizer(
        RelationVerbalizer,
        QuantityVerbalizer,
        MentionAffixes,
        ):
    args = RelationVerbalizer.args + QuantityVerbalizer.args + [
        ('--numprop-paraphrase-idx', dict(type=int, default=0)),
        ]

    @property
    def conf_fields(self):
        return super().conf_fields + [
            'numprop_paraphrase_idx',
            ]

    @cached_property
    def verbalize_object(self):
        return self.verbalize_quantity

    def template(self, instance=None, pred_id=None, paraphrase_idx=None):
        if pred_id is None:
            pred_id = instance['pred_id']
        if paraphrase_idx is None:
            paraphrase_idx = self.conf.numprop_paraphrase_idx
        return self.templates()[pred_id][paraphrase_idx]

    @staticmethod
    def templates():
        return _pred2paraphrase_idx2question_tpl

    @staticmethod
    def prefix_of_template(template):
        return MentionAffixes().prefix(template)

    @staticmethod
    def suffix_of_template(template):
        return MentionAffixes().suffix(template)

    def write_affix_indices(self, data):

        def subw_len(text):
            return data.subw_len(text, ignore_whitespace=True)

        configs = defaultdict(lambda: defaultdict(list))

        for pred_id, paraphrases in _pred2paraphrase_idx2question_tpl.items():
            for paraphrase_idx, _ in enumerate(paraphrases):
                tpl = self.template(
                    pred_id=pred_id, paraphrase_idx=paraphrase_idx)

                for affix_name in 'prefix', 'suffix':
                    affix = getattr(self, affix_name)(tpl)
                    affix_len = subw_len(affix)
                    affix_key = affix_name + '_idx'

                    for idx in range(affix_len):
                        config = {
                            'numprop_paraphrase_idx': paraphrase_idx,
                            'text_enc_pooling': affix_key,
                            affix_key: idx,
                            }
                        configs[pred_id][self.conf.transformer].append(config)

        outfile = self.affix_indices_file(self.conf.transformer)
        json_dump(configs, outfile)
        return outfile

    def affix_indices_file(self, tokenizer_name):
        tokenizer_str = tokenizer_name.replace('/', '_')
        fname = f'verbalizer_indices.{tokenizer_str}.json'
        return Path('configs') / fname

    def affix_indices(self, *, pred_id, tokenizer_name):
        affix_indices_file = self.affix_indices_file(tokenizer_name)
        pred2tok2affix_indices = json_load(affix_indices_file)
        return pred2tok2affix_indices[pred_id][tokenizer_name]

    @cache
    def paraphrase_idx2affix2idx(self, *, pred_id, tokenizer_name):
        from collections import defaultdict
        affix_indices = self.affix_indices(
            pred_id=pred_id, tokenizer_name=tokenizer_name)
        paraphrase_idx2affix2idx = defaultdict(lambda: defaultdict(list))

        for idx_conf in affix_indices:
            paraphrase_idx = idx_conf['numprop_paraphrase_idx']
            affix = idx_conf['text_enc_pooling']
            idx = idx_conf[affix]
            paraphrase_idx2affix2idx[paraphrase_idx][affix].append(idx)
        return paraphrase_idx2affix2idx


class NumericRelationVerbalizer_Question(
        NumericRelationVerbalizer, MentionAffixes):

    @cached_property
    def verbalize_object(self):
        return lambda instance: None


class WithVerbalizer(Configurable):
    args = Configurable.args + [
        ('--numprop-verbalizer', dict(type=str, default='question')),
        ('--add-initial-space', dict(type=bool, default=True)),
        ('--prompt-template', dict(type=str, default='gpt-jt')),
        ]

    @property
    def conf_fields(self):
        return self.verbalizer.conf_fields + [
            'numprop_verbalizer',
            'prompt_template',
            'numprop_paraphrase_idx',
            'add_initial_space',
            ]

    @property
    def verbalizer_conf_str(self):
        return self.conf.numprop_verbalizer

    def verbalizer_cls(self, conf_str):
        return {
            'mention': MentionVerbalizer,
            'question': NumericRelationVerbalizer_Question,
            }[conf_str]

    @cached_property
    def verbalizer(self):
        verbalizer = self.verbalizer_cls(self.verbalizer_conf_str)(self.conf)
        if self.conf.numprop_train_file:
            verbalizer.raw = self.raw
        return verbalizer

    def verbalize(self, instance):
        return self.verbalizer.verbalize(instance)

    def _texts(self, instances):
        texts = list(map(self.verbalize, instances))
        if self.conf.add_initial_space:
            texts = [' ' + text for text in texts]
        return texts
