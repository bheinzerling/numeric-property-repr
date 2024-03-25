from functools import cache
import torch

from collections import Counter

from util import (
    cached_property,
    avg,
    decimal_round,
    )

from transformer_mixin import TransformerLM


class TransformerRepr(TransformerLM):
    args = TransformerLM.args + [
        ('--text-enc-layer', dict(type=int, default=None)),
        ('--text-enc-layer-relative', dict(type=float, default=None)),
        ('--text-enc-pooling', dict(
            type=str,
            choices=[
                'mean_mention',
                'mention_last',
                'mention_first',
                'mention_first_last',
                'after_mention',
                'mask',
                'mean_sent',
                'decoder_mean_sent',
                'decoder_first',
                'prefix_idx',
                'suffix_idx',
                ],
            default='mean_mention')),
        ('--prefix-idx', dict(type=int, default=0)),
        ('--suffix-idx', dict(type=int, default=0)),
        ('--fp16', dict(action='store_true')),
        ('--text-enc-output-device', dict(type=str, default='cpu')),
        ('--text-enc-remove-tensor-keys', dict(type=str, nargs='+', default=['logits'])),
        ('--mention-pooling-inconsistency-tolerance', dict(type=float, default=0.02)),
        ]

    def __init__(self, *args, transformer=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.conf.text_enc_layer is not None and self.conf.text_enc_layer_relative is not None:
            raise ValueError('Cannot set both --text-enc-layer and --text-enc-layer-relative')
        if self.conf.text_enc_pooling.startswith('decoder_'):
            self.conf.trf_include_dec_states = True

    @property
    def transformer_conf_str(self):
        model = self.conf.transformer.split('/')[-1]
        return '.'.join([
            model,
            self.conf.text_enc_pooling + '_pooling',
            ])

    @property
    def conf_fields(self):
        return super().conf_fields + [
            'text_enc_layer',
            'text_enc_layer_relative',
            'text_enc_pooling',
            'fp16',
            'prefix_idx',
            'suffix_idx',
            ]

    @property
    def texts(self):
        raise NotImplementedError()

    @cached_property
    def texts_enc(self):
        if isinstance(self.texts, dict):
            return {
                split_name: self._texts_enc(texts)
                for split_name, texts in self.texts.items()
                }
        return self._texts_enc(self.texts)

    @property
    def n_layers(self):
        return self.n_layers_from_config(self.trf.config)

    @staticmethod
    def n_layers_from_config(config):
        for attr in ['n_layer', 'num_layers', 'num_hidden_layers']:
            try:
                return getattr(config, attr)
            except AttributeError:
                pass
        raise NotImplementedError(
            f'Cannot retrieve number of layers for {config._name_or_path}')

    @cache
    def n_layers_from_model_name(self, model_name):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        return self.n_layers_from_config(config)

    @cached_property
    def text_enc_layer_absolute(self):
        r = self.conf.text_enc_layer_relative
        if r is None:
            return self.conf.text_enc_layer
        layer_absolute = self.layer_rel_to_abs(r, n_layers=self.n_layers)
        self.log(f'relative layer {r} converted to absolute layer {layer_absolute}')
        return layer_absolute

    def layer_rel_to_abs(
            self, layer_rel, n_layers=None, model_name=None):
        assert n_layers is not None or model_name is not None
        if n_layers is None:
            n_layers = self.n_layers_from_model_name(model_name)
        # the hidden_states tensor returned by transformers models has
        # (n_layers + 1 embedding layer) in the layer-dimension.
        # n_layers does not count the embedding layer, but since Python is 0-indexed
        # this works out exactly so that 0 is the index into the embedding layer and
        # self.n_layers the index for the last layer
        layer_fractional = layer_rel * n_layers
        layer_absolute = int(decimal_round(layer_fractional))
        return layer_absolute

    def is_valid_layer_idx_range(
            self,
            *,
            layer_idx,
            left_window_size,
            right_window_size,
            model_name,
            ):
        assert left_window_size >= 0
        if layer_idx - left_window_size < 0:
            return False
        n_layers = self.n_layers_from_model_name(model_name)
        if layer_idx + right_window_size > n_layers:
            return False
        return True

    @cached_property
    def text_enc_layer_relative(self):
        return self.layer_abs_to_rel(
            self.text_enc_layer_absolute, n_layers=self.n_layers)

    def layer_abs_to_rel(
            self, layer_idx_abs, n_layers=None, model_name=None):
        assert n_layers is not None or model_name is not None
        if n_layers is None:
            n_layers = self.n_layers_from_model_name(model_name)
        layer_fractional = layer_idx_abs / n_layers
        return decimal_round(layer_fractional, 2)

    def _texts_enc(self, texts):
        return self.encode_texts(
            texts,
            output_fp16=self.conf.fp16,
            output_device=self.conf.text_enc_output_device,
            hidden_states_layer=self.text_enc_layer_absolute,
            remove_tensor_keys=self.conf.text_enc_remove_tensor_keys,
            )

    def subw_len(
            self,
            text,
            disregard_initial_space=False,
            disregard_final_space=False,
            add_initial_space=False,
            ignore_whitespace=False,
            ):
        if add_initial_space:
            text = ' ' + text
        if disregard_final_space:
            text = text.rstrip()
        if ignore_whitespace:
            text = text.strip()
            return min(
                self._subw_len(text, disregard_initial_space=True),
                self._subw_len(' ' + text, disregard_initial_space=True)
                )
        return self._subw_len(
            text, disregard_initial_space=disregard_initial_space)

    def _subw_len(self, text, disregard_initial_space=False):
        tokenized = self.tokenizer.tokenize(text, add_special_tokens=False)
        space = self.tokenizer.tokenize(' ')[0]
        if disregard_initial_space and tokenized and tokenized[0] == space:
            len_correction = 1
        else:
            len_correction = 0
        return len(tokenized) - len_correction

    @cached_property
    def repr_dim(self):
        repr_dim = self.trf_config.hidden_size
        if self.conf.text_enc_pooling == 'mention_first_last':
            repr_dim *= 2
        return repr_dim

    @cached_property
    def text_reprs(self):
        return self._text_reprs(self.texts_enc)

    def _text_reprs(self, texts_enc):
        return self.pool(texts_enc)

    def get_hidden_states(self, enc_out):
        if self.conf.text_enc_pooling.startswith('decoder'):
            return enc_out['decoder_hidden_states']
        try:
            return enc_out['hidden_states']
        except KeyError:
            return enc_out['encoder_hidden_states']

    @property
    def hidden_state_layer_dim(self):
        return 1

    def selection_idx(self, enc_out, instances=None):
        select_fn_name = f'selection_idx_{self.conf.text_enc_pooling}'
        return getattr(self, select_fn_name)(enc_out, instances=instances)

    def pool(self, enc_out, instances=None):
        pool_fn_name = 'pool_' + self.conf.text_enc_pooling
        pool_fn = getattr(self, pool_fn_name)
        hidden_pooled = pool_fn(enc_out, instances=instances)
        if self.one_layer_only:
            # check that the transformer mixin has return a single layer
            assert hidden_pooled.shape[self.hidden_state_layer_dim] == 1, breakpoint()
            hidden_pooled = hidden_pooled.squeeze(self.hidden_state_layer_dim)
        return hidden_pooled

    def get_content_mask(self, enc_out):
        """Returns a tensor whose entries are 1 for 'content' input ids
        and zero for PAD, BOS, and EOS tokens
        """
        attn_mask = enc_out['attention_mask']
        pool_mask = attn_mask.clone()
        # set postions of BOS and EOS tokens to 0, since we don't want to
        # include those in the pooled representation
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        for token_id in (bos_id, eos_id):
            if token_id is not None:
                pool_mask[enc_out['input_ids'] == token_id] = 0
        # set postions of EOS tokens to 0, since we don't want to include it
        if self.eos_offset == 1:
            pool_mask.scatter_(1, attn_mask.sum(dim=1).unsqueeze(1) - 1, 0)
        return pool_mask

    def get_mention_mask(self, enc_out, *, instances, mode='all'):
        attn_mask = enc_out['attention_mask']
        assert instances is not None
        prefixes = self.verbalizer.prefixes(instances)
        mentions = self.verbalizer.mentions(instances)
        prefix_len_list = [self.subw_len(prefix, ignore_whitespace=True) for prefix in prefixes]
        prefix_len = torch.Tensor(prefix_len_list).unsqueeze(1).long()
        mention_len = torch.Tensor([
            self.subw_len(m, disregard_final_space=True, add_initial_space=self.conf.add_initial_space) for m in mentions
            ]).unsqueeze(1).long()

        if self.is_pad_left:
            assert attn_mask[:, -1].all()
            left_pad_lengths = (~(attn_mask.bool())).sum(dim=1).unsqueeze(1)
        else:
            assert attn_mask[:, 0].all()
            left_pad_lengths = torch.zeros_like(prefix_len)

        mention_start_idx = left_pad_lengths + self.bos_offset + prefix_len
        mention_end_idx = mention_start_idx + mention_len  # end exclusive
        assert (mention_start_idx < mention_end_idx).all(), breakpoint()
        pool_mask = self.get_content_mask(enc_out)
        all_idxs = torch.arange(pool_mask.size(1)).unsqueeze(0)
        seq_len = attn_mask.shape[-1]

        if mode == 'all':
            pre_mention_mask = all_idxs < mention_start_idx
            post_mention_mask = all_idxs >= mention_end_idx
            pool_mask[pre_mention_mask] = 0
            pool_mask[post_mention_mask] = 0
            assert pool_mask.sum(dim=1).min() > 0, breakpoint()
            # mention_len = lengths - prefix_len - suffix_len
            pooling_is_inconsistent = (pool_mask.sum(dim=1) != mention_len.squeeze(1))
            if pooling_is_inconsistent.any():
                inconsistency = pooling_is_inconsistent.float().mean()
                n_inconsistent = pooling_is_inconsistent.sum()
                self.log(f'mention pooling inconsistency: {n_inconsistent} / {len(mentions)} = {inconsistency:.4f}')
                assert inconsistency < self.conf.mention_pooling_inconsistency_tolerance
        elif mode == 'last':
            mention_end_idx.clamp_(max=seq_len)
            # mention_end_idx is end-exclusive
            mention_last_idx = mention_end_idx - 1
            # TODO: refactor, this is not really a mask
            pool_mask = mention_last_idx.squeeze(-1)
        elif mode == 'after':
            mention_end_idx.clamp_(max=seq_len)
            # mention_end_idx is end-exclusive
            after_mention_idx = mention_end_idx
            # TODO: refactor, this is not really a mask
            pool_mask = after_mention_idx.squeeze(-1)
        elif mode == 'first':
            pool_mask = mention_start_idx.squeeze(-1)
        else:
            raise NotImplementedError(mode)
        return pool_mask

    def pool_mean(self, enc_out, mention_only=True, instances=None):
        """mean pooling over token representation (along the sequence axis)."""
        hidden_states = self.get_hidden_states(enc_out)
        if mention_only:
            pool_mask = self.get_mention_mask(enc_out, instances=instances)
        else:
            pool_mask = self.get_content_mask(enc_out)

        # retain only the all hidden states corresponding the content tokens,
        # set all hidden states corresponding to BOS, EOS, or PAD tokens to 0
        hidden_masked = hidden_states * pool_mask.unsqueeze(1).unsqueeze(-1)
        # the sequence lengths of all inputs, excluding BOS and EOS tokens
        lengths = pool_mask.sum(dim=1, keepdim=True)
        assert lengths.min() > 0, breakpoint()
        # mean pooling along the sequence axis
        # shape: (n_reprs x layers x n_hidden)
        hidden_pooled = hidden_masked.sum(dim=2) / lengths.view(-1, 1, 1)

        return hidden_pooled

    def pool_encoder_mean(self, enc_out, instances=None):
        breakpoint()

    def pool_mean_mention(self, enc_out, instances=None):
        return self.pool_mean(enc_out, mention_only=True, instances=instances)

    def selection_idx_mention_last(self, enc_out, instances=None):
        return self.get_mention_mask(enc_out, instances=instances, mode='last')

    def pool_mention_last(self, enc_out, instances=None):
        idx = self.selection_idx_mention_last(enc_out, instances=instances)
        return self.take_hidden_state(enc_out, idx)

    def selection_idx_mention_first(self, enc_out, instances=None):
        return self.get_mention_mask(enc_out, instances=instances, mode='first')

    def pool_mention_first(self, enc_out, instances=None):
        idx = self.selection_idx_mention_first(enc_out, instances=instances)
        return self.take_hidden_state(enc_out, idx)

    def pool_mention_first_last(self, enc_out, instances=None):
        first = self.pool_mention_first(enc_out, instances=instances)
        last = self.pool_mention_last(enc_out, instances=instances)
        return torch.concat([first, last], dim=-1)

    def selection_idx_after_mention(self, enc_out, instances=None):
        return self.get_mention_mask(enc_out, instances=instances, mode='after')

    def pool_after_mention(self, enc_out, instances=None):
        idx = self.selection_idx_after_mention(enc_out, instances=instances)
        return self.take_hidden_state(enc_out, idx)

    def selection_idx_prefix_idx(self, enc_out, instances=None):
        if self.is_pad_left:
            pad_mask = enc_out['input_ids'] == self.tokenizer.pad_token_id
            offset = pad_mask.sum(dim=1)
        else:
            offset = torch.zeros_like(enc_out['input_ids'][:, 0])
        idx = offset + self.conf.prefix_idx
        sequence_dim = 1
        prefix_token_ids = enc_out['input_ids'].take_along_dim(
            indices=idx.view(-1, 1), dim=sequence_dim
            ).squeeze(sequence_dim)
        assert len(set(prefix_token_ids.tolist())) == 1
        return idx

    def pool_prefix_idx(self, enc_out, instances=None):
        idx = self.selection_idx_prefix_idx(enc_out, instances=instances)
        return self.take_hidden_state(enc_out, idx)

    def selection_idx_suffix_idx(self, enc_out, instances=None):
        mention_last_idx = self.get_mention_mask(enc_out, instances=instances, mode='last')
        suffix_first_idx = mention_last_idx + 1
        idx = suffix_first_idx + self.conf.suffix_idx

        sequence_dim = 1
        sequence_len = enc_out['input_ids'].shape[sequence_dim]

        # in rare cases (see comment below) the last token of the mention can get
        # merged with the first token of the suffix. This leads to an index error
        # if this merged mention-suffix token is also the last token of the input
        idx.clamp_(max=sequence_len - 1).min()

        suffix_token_ids = enc_out['input_ids'].take_along_dim(
            indices=idx.view(-1, 1), dim=sequence_dim
            ).squeeze(sequence_dim)
        suffix_token_ids = suffix_token_ids.tolist()
        if len(set(suffix_token_ids)) != 1:
            counts = Counter(suffix_token_ids)
            top_count = counts.most_common()[0][1]
            # there are some cases in which the last part of the mention is tokenized as part of the suffix, e.g.:
            # self.subw_len(' What is the birthyear of Benjamin Silliman, Sr.?') -> 13
            # self.subw_len(' What is the birthyear of Benjamin Silliman, Sr.') -> 13
            # the last token in the first case is '.?'
            # we'll ignore these cases if there are not too many (less than 5%) of them
            assert top_count / len(suffix_token_ids) > 0.95, breakpoint()
        return idx

    def pool_suffix_idx(self, enc_out, instances=None):
        idx = self.selection_idx_suffix_idx(enc_out, instances=instances)
        return self.take_hidden_state(enc_out, idx)

    def take_hidden_state(self, enc_out, idx):
        hidden_states = self.get_hidden_states(enc_out)
        sequence_dim = 2
        pooled = torch.take_along_dim(
            input=hidden_states,
            indices=idx.view(-1, 1, 1, 1),
            dim=sequence_dim,
            )
        return pooled.squeeze(sequence_dim)

    def pool_mean_sent(self, enc_out, instances=None):
        return self.pool_mean(enc_out, mention_only=False, instances=instances)

    def pool_decoder_mean_sent(self, enc_out, instances=None):
        return self.pool_mean(enc_out, mention_only=False, instances=instances)

    def pool_decoder_first(self, enc_out, instances=None):
        seq_len = enc_out['attention_mask'].sum(dim=1)
        # in causal LMs, generatation is conditioned on the hidden
        # states of all tokens generated so far. The first token
        # when nothing has been generated so far is the last token
        # of the input
        idx = seq_len
        return self.take_hidden_state(enc_out, idx)

    def pool_mask(self, enc_out, instances=None):
        hidden_states = self.get_hidden_states(enc_out)
        mask = enc_out['input_ids'] == self.tokenizer.mask_token_id
        hidden_masked = hidden_states * mask.float().unsqueeze(1).unsqueeze(-1)
        hidden_pooled = hidden_masked.sum(dim=-2)
        hidden_pooled = hidden_pooled.permute(1, 0, 2)
        return hidden_pooled

    @property
    def one_layer_only(self):
        return self.text_enc_layer_absolute is not None

    @cached_property
    def prompt_model_preds(self):
        if '<mask>' in self.conf.verbalizer:
            mask = self.texts_enc['input_ids'] == self.tokenizer.mask_token_id
            token_ids = self.texts_enc['logits'][mask].argmax(dim=-1)
            tokens = self.tokenizer.batch_decode(token_ids)
            return [t.replace(' ', '') for t in tokens]

    @cached_property
    def prompt_pred_acc(self):
        preds = self.prompt_model_preds
        if len(preds) != len(self.raw):
            breakpoint()
        if preds is not None:
            for i, inst in enumerate(self.raw):
                inst['text'] = self.texts[i]
                inst['answer'] = getattr(self, 'answer' + inst['class'])
                inst['model_pred'] = preds[i]
                inst['correct'] = int(inst['answer'] == inst['model_pred'])
            acc = avg([inst['correct'] for inst in self.raw])
            return acc

    def inputs_to_device(self, inputs):
        device = self.trf.device
        return inputs.to(device)
