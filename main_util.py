from functools import cache
from types import SimpleNamespace

from util import (
    cached_property,
    )


class MainUtil:
    @cached_property
    def quantity_parser(self):
        from quantity_parser import QuantityParser
        return QuantityParser()

    @cache
    def parse_value(self, text, default_value=0):
        value = self.quantity_parser(text)
        try:
            return self.data.number_type_hint(value)
        except Exception:
            return default_value

    def lm_output(self, prompt, prompt_len=None, do_log=False):
        if prompt_len is None:
            prompt_len = len(prompt)
        output_raw = self.data.generate(
            prompt,
            do_sample=False,
            max_new_tokens=self.conf.edit_max_new_tokens,
            )[0]
        output_raw = output_raw[prompt_len:].replace('\n', '\\n')
        output_parsed = self.parse_value(output_raw)
        if do_log:
            self.log(f'{prompt} -> {output_raw} [parsed: {output_parsed}]')
        return SimpleNamespace(
            raw=output_raw,
            parsed=output_parsed,
            )
