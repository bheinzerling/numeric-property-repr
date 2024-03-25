import torch

from util import longest_prefix_match


def hidden_states_hook(func):
    def wrapper(self, module, args, kwargs, output):
        if isinstance(output, torch.Tensor):
            hidden_states = output
            return_tensor = True
        else:
            assert isinstance(output, tuple)
            hidden_states = output[0]
            return_tensor = False
        new_hidden_states = func(self, module, args, kwargs, hidden_states)
        if return_tensor:
            new_output = new_hidden_states
        else:
            new_output = (new_hidden_states, ) + output[1:]
        return new_output
    return wrapper


class ForwardHook:
    """Base class for forward hooks. The main thing this class does
    is, if necessary, to automatically remove itself before attaching
    """
    def __init__(self, model):
        self.model = model

    def attach(self, hook_func, attachment_point):
        if hasattr(self, 'handle') and self.handle is not None:
            self.remove()
        self.handle = attachment_point.register_forward_hook(
                hook_func, with_kwargs=True)

    def remove(self):
        self.handle.remove()


class LayerForwardHook(ForwardHook):
    """A forward hook that attaches to a particular Transformer LM layer,
    specified by layer_idx [0, num_layers], where 0 represents the
    model's (sub)word embedding layer and num_layers the last layer.
    """
    def attach(self, hook_func, layer_idx):
        self.layer_idx = layer_idx
        attachment_point = self.layers[layer_idx]
        super().attach(hook_func, attachment_point)

    @property
    def emb_key(self):
        return 'emb'

    @property
    def non_emb_key(self):
        return 'non_emb'

    @property
    def model_family2layer_funcs(self):
        emb = self.emb_key
        non_emb = self.non_emb_key
        return {
            'BLOOM': {
                emb: lambda model: model.transformer.word_embeddings,
                non_emb: lambda model: list(model.transformer.h),
                },
            'Falcon': {
                emb: lambda model: model.transformer.word_embeddings,
                non_emb: lambda model: list(model.transformer.h),
                },
            'GPTJ': {
                emb: lambda model: model.transformer.wte,
                non_emb: lambda model: list(model.transformer.h),
                },
            'Llama-2': {
                emb: lambda model: model.model.embed_tokens,
                non_emb: lambda model: list(model.model.layers),
                },
            'Mistral': {
                emb: lambda model: model.model.embed_tokens,
                non_emb: lambda model: list(model.model.layers),
                },
        }

    @property
    def model_families(self):
        return list(self.model_family2layer_funcs)

    @property
    def model_family(self):
        return longest_prefix_match(
            self.model.__class__.__name__, self.model_families)

    @property
    def layers(self):
        return [self.emb_layer] + self.non_emb_layers

    @property
    def n_layers(self):
        return len(self.layers)

    def layer_funcs(self, key):
        return self.model_family2layer_funcs[self.model_family][key]

    @property
    def emb_layer(self):
        layer_func = self.layer_funcs(self.emb_key)
        return layer_func(self.model)

    @property
    def non_emb_layers(self):
        layer_func = self.layer_funcs(self.non_emb_key)
        return layer_func(self.model)


@hidden_states_hook
def zero_activation(module, args, kwargs, output):
    return torch.zeros_like(output)


class OffsetActivation:
    def __init__(self, offset):
        self.offset = offset

    @hidden_states_hook
    def __call__(self, module, args, kwargs, output):
        return output + self.offset


class RandomActivation:
    def __init__(self, abs_scale=None, std_scale=None):
        self.abs_scale = abs_scale
        self.std_scale = std_scale
        if abs_scale is not None:
            assert std_scale is None
            self.scale = abs_scale
        if std_scale is not None:
            assert abs_scale is None
            self.scale = std_scale

    @hidden_states_hook
    def __call__(self, module, args, kwargs, output):
        return self.noise_like(output)

    def noise_like(self, output):
        noise = torch.randn(*output.size()).to(output)
        if self.std_scale is None:
            std = 1
        else:
            std = output.std()
        self.std = std
        return noise * std * self.scale


class RandomOffsetActivation(RandomActivation):
    @hidden_states_hook
    def __call__(self, module, args, kwargs, output):
        return output + self.noise_like(output)


class DirectedEdit:
    def __init__(self, *, direction, token_idx, seq_dim=1, call_once=False):
        self.direction = direction
        self.token_idx = token_idx
        self.seq_dim = seq_dim
        self.call_once = call_once
        self.call_again = True

    @hidden_states_hook
    def __call__(self, module, args, kwargs, output):
        if not self.call_again:
            return output
        if self.call_once:
            self.call_again = False
        direction = torch.zeros_like(output)
        direction = direction.transpose(0, self.seq_dim)
        direction[self.token_idx] = self.direction.to(output)
        direction = direction.transpose(0, self.seq_dim)
        return output + direction


def inspect(module, args, kwargs, output):
    breakpoint()
    return output
