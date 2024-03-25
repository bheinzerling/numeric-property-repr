import functools

from util import cached_property


class QuantityParser:
    @cached_property
    def number_pattern(self):
        import re
        return re.compile('-?[0-9,.]+')

    @property
    def default_value(self):
        return 0

    @cached_property
    def parse_with_parser(self):
        from quantulum3 import parser
        from pint import UnitRegistry
        import traceback

        ureg = UnitRegistry()

        def parse_geocoord(text):
            quantities = parser.parse(text)
            raise NotImplementedError('TODO: improve geo-coord parsing')

        def parse(
                text,
                default_on_error=False,
                error_sentinel=None,
                expect_geocoord=False,
                ):
            text = text.split('\\n')[0]
            text = text.strip('.')

            if expect_geocoord:
                try:
                    magnitude = parse_geocoord(text)
                except Exception:
                    pass
                if magnitude is not None:
                    return magnitude

            try:
                quantity = ureg.Quantity(text)
            except Exception:
                try:
                    # "unit" BC (before Christ) causes an exception in
                    # quantulum. Workaround: turn the string into a negative
                    # year
                    if text.lower().endswith('bc'):
                        text = '-' + text[:-2]
                    quantities = parser.parse(text)
                except Exception:
                    traceback.print_exc()
                    quantities = None
                if not quantities:
                    return error_sentinel
                quantity = quantities[0]
                if quantity.unit.name == 'dimensionless':
                    quantity_str = str(quantity.value)
                else:
                    quantity_str = f'{quantity.value} {quantity.unit.name}'
                try:
                    quantity = ureg.Quantity(quantity_str)
                except Exception:
                    if default_on_error:
                        return self.default_value
                    else:
                        return error_sentinel

            # convert time values to years
            if '[time]' in set((quantity.dimensionality.keys())):
                quantity = quantity.to_preferred([ureg.Unit('year')])

            magnitude = quantity.m
            return magnitude

        return parse

    @functools.cache
    def __call__(
            self,
            text,
            default_on_error=False,
            error_sentinel=None,
            expect_geocoord=False,
            ):
        value = self.parse_with_parser(
            text,
            default_on_error=False,
            error_sentinel=error_sentinel,
            )
        if value is not error_sentinel:
            return value
        return self.parse_with_regex(
            text,
            default_on_error=default_on_error,
            error_sentinel=error_sentinel,
            )

    def parse_with_regex(
            self, text, default_on_error=False, error_sentinel=None):
        match = self.number_pattern.search(text.strip())
        if match:
            num_str = match.group().lstrip(',.').rstrip(',')
        else:
            num_str = ''
        try:
            return float(num_str)
        except ValueError:
            if default_on_error:
                return self.default_value
            else:
                return error_sentinel

    def parse_with_default(self, text):
        return self(text, default_on_error=True)

    def preds_error_mask(self, raw_preds):
        sentinel = object()
        return [
            self.parse_with_parser(pred, error_sentinel=sentinel) == sentinel
            for pred in raw_preds
            ]
