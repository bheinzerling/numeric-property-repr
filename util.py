import re
import logging
from pathlib import Path
import random
from argparse import ArgumentParser
import json
import six
from collections import defaultdict
from collections.abc import Iterable

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from seaborn.utils import relative_luminance


class _Missing(object):

    def __repr__(self):
        return 'no value'

    def __reduce__(self):
        return '_missing'


_missing = _Missing()

_global_cache = dict()


class _cached_property(property):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value:
        class Foo(object):
            @cached_property
            def foo(self):
                # calculate something important here
                return 42
    The class has to have a `__dict__` in order for this property to
    work.
    """
    # source: https://github.com/pallets/werkzeug/blob/master/werkzeug/utils.py

    # implementation detail: A subclass of python's builtin property
    # decorator, we override __get__ to check for a cached value. If one
    # choses to invoke __get__ by hand the property will still work as
    # expected because the lookup logic is replicated in __get__ for
    # manual invocation.

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __set__(self, obj, value):
        obj.__dict__[self.__name__] = value

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value

    def __delete__(self, obj):
        del obj.__dict__[self.__name__]

    # https://www.ianlewis.org/en/pickling-objects-cached-properties
    def __getstate__(self):
        state = self.__dict__.copy()
        for key in state:
            if (hasattr(self.__class__, key) and
                    isinstance(getattr(self.__class__, key), _cached_property)):
                pass
                # del state[key]
                # state[key] = self.func(key)
        return state


def cached_property(func=None, **kwargs):
    # https://stackoverflow.com/questions/7492068/python-class-decorator-arguments
    if func:
        return _cached_property(func)
    else:
        def wrapper(func):
            return _cached_property(func, **kwargs)

        return wrapper


def avg(values):
    n_values = len(values)
    if n_values:
        return sum(values) / n_values
    return 0


def decimal_round(number, n_decimal_places=0):
    from decimal import Decimal, ROUND_HALF_UP
    round_fmt = '1'
    if n_decimal_places > 0:
        round_fmt += '.' + '0' * n_decimal_places
    return Decimal(number).quantize(Decimal(round_fmt), rounding=ROUND_HALF_UP)


def make_rundir(basedir=Path("out"), runid_fname="runid", log=None):
    """Create a directory for running an experiment."""
    import uuid
    runid = str(uuid.uuid4())
    rundir = basedir / str(runid)
    rundir.mkdir(exist_ok=True, parents=True)
    if log:
        log(f"rundir: {rundir.resolve()}")
    return rundir


def make_and_set_rundir(conf, log=None):
    """Make rundir and set conf.rundir to the corresponding path."""
    if getattr(conf, 'runid', None) is not None:
        conf.rundir = mkdir(conf.outdir / conf.runid)
    else:
        conf.rundir = make_rundir(log=log)
        conf.runid = conf.rundir.name


def longest_prefix_match(prefix, targets):
    from os.path import commonprefix
    max_len = 0
    longest_match = None
    for target in targets:
        match = commonprefix([prefix, target])
        match_len = len(match)
        if match_len > max_len:
            max_len = match_len
            longest_match = target
    return longest_match


def precision_and_type_for_pred_and_unit(pred_id, unit_id):
    date_of_birth = 'P569'
    date_of_death = 'P570'
    inception = 'P571'
    dissolved_date = 'P576'
    publication_date = 'P577'
    point_in_time = 'P585'
    latitude = 'P625.lat'
    longitude = 'P625.long'
    population = 'P1082'
    number_of_households = 'P1538'
    work_period_start = 'P2031'
    elevation = 'P2044'
    area = 'P2046'
    duration = 'P2047'

    annum = 'Q1092296'
    metre = 'Q11573'
    square_km = 'Q712226'
    minute = 'Q7727'
    degree = 'Q28390'
    no_unit = '1'

    pred_id_and_unit_id2precision_and_type = {
        (work_period_start, annum): (0, int),
        (duration, minute): (0, int),
        (date_of_birth, annum): (0, int),
        (date_of_death, annum): (0, int),
        (inception, annum): (0, int),
        (dissolved_date, annum): (0, int),
        (publication_date, annum): (0, int),
        (point_in_time, annum): (0, int),
        (population, no_unit): (0, int),
        (number_of_households, no_unit): (0, int),
        (elevation, metre): (0, int),
        }
    return pred_id_and_unit_id2precision_and_type.get(
            (pred_id, unit_id), (None, None))


# https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph  # NOQA
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    from mpl_toolkits import axes_grid1
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def simple_imshow(
        matrix,
        cmap="viridis",
        figsize=(10, 10),
        aspect_equal=True,
        outfile=None,
        title=None,
        xlabel=None,
        ylabel=None,
        xticks=True,
        yticks=True,
        xtick_labels=None,
        ytick_labels=None,
        xtick_locs_labels=None,
        ytick_locs_labels=None,
        tick_labelsize=None,
        xtick_label_rotation='vertical',
        xgrid=None,
        ygrid=None,
        colorbar=True,
        scale="lin",
        colorbar_range=None,
        cbar_title=None,
        bad_color='white',
        origin='upper',
        cell_text=None,
        cell_text_color=None,
        ):
    if aspect_equal and figsize is not None and figsize[1] is None:
        matrix_aspect = matrix.shape[0] / matrix.shape[1]
        width = figsize[0]
        height = max(3, width * matrix_aspect)
        figsize = (width, height)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if aspect_equal:
        ax.set_aspect('equal')
    if title:
        plt.title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    norm = mpl.colors.SymLogNorm(1) if scale == "log" else None
    cmap = mpl.cm.get_cmap(cmap)
    if bad_color is not None:
        cmap.set_bad(bad_color)
    im = plt.imshow(
        matrix, interpolation='nearest', cmap=cmap, norm=norm, origin=origin)
    if xtick_labels is not None:
        assert xtick_locs_labels is None
        locs = np.arange(0, len(xtick_labels))
        xtick_locs_labels = locs, xtick_labels
    if ytick_labels is not None:
        assert ytick_locs_labels is None
        locs = np.arange(0, len(ytick_labels))
        ytick_locs_labels = locs, ytick_labels
    if xtick_locs_labels is not None:
        plt.xticks(*xtick_locs_labels, rotation=xtick_label_rotation)
    if ytick_locs_labels is not None:
        plt.yticks(*ytick_locs_labels)
    if xgrid is not None or ygrid is not None:
        if xgrid is not None:
            ax.set_xticks(xgrid, minor=True)
        if ygrid is not None:
            ax.set_yticks(ygrid, minor=True)
        ax.grid(which="minor")
    if xticks is not True:
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,)         # ticks along the top edge are off
        ax.set_xticks([])
    if yticks is not True:
        plt.tick_params(
            axis='y',          # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,)         # ticks along the top edge are off
        ax.set_yticks([])
    if colorbar:
        cbar = add_colorbar(im)
        if colorbar_range is not None:
            plt.clim(*colorbar_range)
        if cbar_title:
            cbar.ax.set_ylabel(cbar_title, labelpad=3, rotation=90)
    if tick_labelsize is not None:
        ax.xaxis.set_tick_params(labelsize=tick_labelsize)
        ax.yaxis.set_tick_params(labelsize=tick_labelsize)
    if cell_text is not None:
        for (i, j), val in np.ndenumerate(matrix):
            color = cmap(val)
            # source: https://github.com/mwaskom/seaborn/blob/6890b315d00b74f372bc91f3929c803837b2ddf1/seaborn/matrix.py#L258
            lum = relative_luminance(color)
            if cell_text_color is None:
                _cell_text_color = ".15" if lum > .408 else "w"
            else:
                _cell_text_color = cell_text_color
            ax.text(j, i, cell_text[i, j], color=_cell_text_color, ha='center', va='center')
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
    plt.clf()


class Figure:
    """Provides a context manager that automatically saves and closes
    a matplotlib plot.

    >>> with Figure("figure_name"):
    >>>     plt.plot(x, y)
    >>> # saves plot to {Figure.fig_dir}/{figure_name}.{Figure.file_type}

    When creating many figures with the same settings, e.g. plt.xlim(0, 100)
    and plt.ylim(0, 1.0), defaults can be set with:

    >>> Figure.set_defaults(xlim=(0, 100), ylim=(0, 1.0))
    >>> # context manager will call plt.xlim(0, 100) and plt.ylim(0, 1.0)
    """
    fig_dir = Path("out/fig")
    file_types = ["png", "pdf"]
    default_plt_calls = {}
    late_calls = ["xscale", "xlim", "yscale", "ylim"]  # order is important

    def __init__(
            self,
            name,
            figwidth=None,
            figheight=None,
            fontsize=12,
            invert_xaxis=False,
            invert_yaxis=False,
            out_dir=None,
            tight_layout=True,
            savefig_kwargs=None,
            **kwargs,
            ):
        self.fig = plt.figure()
        if figwidth is not None:
            self.fig.set_figwidth(figwidth)
            phi = 1.6180
            self.fig.set_figheight(figheight or figwidth / phi)
        # params = {
        #     'figure.figsize': (figwidth, figheight or figwidth / phi),
        #     'axes.labelsize': fontsize,
        #     'axes.titlesize': fontsize,
        #     'legend.fontsize': fontsize,
        #     'xtick.labelsize': fontsize - 1,
        #     'ytick.labelsize': fontsize - 1,
        # }
        # mpl.rcParams.update(params)
        self.name = name
        self.plt_calls = {**kwargs}
        self.invert_xaxis = invert_xaxis
        self.invert_yaxis = invert_yaxis
        self.tight_layout = tight_layout
        self.savefig_kwargs = savefig_kwargs or {}
        self._out_dir = out_dir
        for attr, val in self.default_plt_calls.items():
            if attr not in self.plt_calls:
                self.plt_calls[attr] = val

    def __enter__(self):
        for attr, val in self.plt_calls.items():
            if attr in self.late_calls:
                continue
            try:
                getattr(plt, attr)(val)
            except:
                getattr(plt, attr)(*val)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for attr in self.late_calls:
            if attr in self.plt_calls:
                getattr(plt, attr)(self.plt_calls[attr])
        if self.invert_xaxis:
            plt.gca().invert_xaxis()
        if self.invert_yaxis:
            plt.gca().invert_yaxis()
        if self.tight_layout:
            plt.tight_layout()
        for file_type in self.file_types:
            fname = f"{self.name}.{file_type}".replace(' ', '_')
            outfile = self.out_dir / fname
            print(outfile, self.savefig_kwargs)
            plt.savefig(outfile, **self.savefig_kwargs)
        plt.clf()

    @classmethod
    def set_defaults(cls, **kwargs):
        cls.default_plt_calls = kwargs
        for attr, val in kwargs.items():
            setattr(cls, attr, val)

    @classmethod
    def reset_defaults(cls):
        cls.default_plt_calls = {}

    @property
    def out_dir(self):
        return self._out_dir or self.fig_dir


class Configurable():
    classes = set()
    args = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        Configurable.classes.add(cls)

    def __init__(self, conf=None, *args, **kwargs):
        super().__init__()
        if conf is None:
            conf = self.get_conf()
        if conf is not None and kwargs:
            from copy import deepcopy
            conf = deepcopy(conf)
        for k, v in kwargs.items():
            if k in conf.__dict__:
                conf.__dict__[k] = v
        self.conf = conf

    def arg_keys(self):
        return [
            arg[0][2:].replace('-', '_') for arg in getattr(self, 'args', [])
        ]

    @property
    def all_conf_fields(self):
        fields = self.conf_fields
        added = set(fields)
        for cls in self.__class__.__mro__:
            if cls is Configurable or cls is self.__class__:
                continue
            if issubclass(cls, Configurable):
                cls_conf_fields = super(cls, self).conf_fields
                for field in cls_conf_fields:
                    if field not in added:
                        added.add(field)
                        fields.append(field)
        return fields

    @property
    def conf_fields(self):
        return []

    @property
    def conf_str(self):
        return conf_hash(self.conf, self.all_conf_fields)

    def conf_str_for_fields(self, fields):
        return '.'.join([
            field + str(getattr(self.conf, field))
            for field in fields])

    @staticmethod
    def get_conf(desc='TODO'):
        a = Configurable.get_argparser(desc=desc)
        args = a.parse_args()
        return args

    @staticmethod
    def get_argparser(desc='TODO'):
        return AutoArgParser(description=desc)

    @staticmethod
    def parse_conf_dict(_dict):
        from types import SimpleNamespace
        a = Configurable.get_argparser()
        dest2action = {action.dest: action for action in a._get_optional_actions()}

        def parse_value(dest, value):
            if value is None:
                return value
            action = dest2action.get(dest, None)
            if action is None:
                return value
            ty = action.type
            if ty is None:
                return value
            if action.nargs in {'+', '*'}:
                return list(map(ty, value))
            return ty(value)

        return SimpleNamespace(**{
            dest: parse_value(dest, value)
            for dest, value in _dict.items()
            })


class AutoArgParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        added_names = dict()
        for cls in Configurable.classes:
            for arg in getattr(cls, 'args', []):
                name, kwargs = arg
                if name in added_names:
                    other_cls, other_kwargs = added_names[name]
                    if kwargs != other_kwargs:
                        raise ValueError(
                            f'Argument conflict. Argument "{name}" exists '
                            f'in {other_cls} with options {other_kwargs} '
                            f'and in {cls} with options {kwargs}')
                    else:
                        continue
                self.add_argument(name, **kwargs)
                added_names[name] = (cls, kwargs)


class EntryPoint(Configurable):
    args = Configurable.args + [
        ('command', dict(type=str, nargs='?')),
    ]

    def run(self):
        getattr(self, self.conf.command)()


class WithRandomSeed(Configurable):
    args = Configurable.args + [
        ('--random-seed', dict(type=int, default=2)),
    ]

    @property
    def conf_fields(self):
        return super().conf_fields + [
            'random_seed',
        ]

    def __init__(self, *args, random_seed=None, **kwargs):
        super().__init__(*args, **kwargs)
        if random_seed is None:
            random_seed = self.conf.random_seed
        self.random_seed = random_seed
        self.set_random_seed(random_seed)

    def set_random_seed(self, seed):
        import numpy as np
        import torch

        self.random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class WithRandomState(WithRandomSeed):

    def set_random_seed(self, seed):
        # do not set any global random seed
        pass

    @cached_property
    def random_state(self):
        return random.Random(self.random_seed)

    @cached_property
    def numpy_random_state(self):
        from numpy.random import RandomState
        return RandomState(self.random_seed)

    @cached_property
    def numpy_rng(self):
        import numpy.random
        return numpy.random.default_rng(self.random_seed)

    @cached_property
    def pytorch_random_state(self):
        import torch
        rng = torch.Generator()
        rng.manual_seed(self.random_seed)
        return rng

    def sample(self, items, sample_size):
        import torch
        rng = self.pytorch_random_state
        rnd_idxs = torch.randperm(len(items), generator=rng)[:sample_size]
        if isinstance(items, torch.Tensor):
            sample = items[rnd_idxs]
        else:
            sample = list(map(items.__getitem__, rnd_idxs))
        assert len(sample) == sample_size
        return sample


def conf_hash(conf, fields=None):
    """Return a hash value for the a configuration object, e.g. an
    argparser instance. Useful for creating unique filenames based on
    the given configuration."""
    if isinstance(conf, dict):
        d = conf
    else:
        if fields is None:
            d = conf.__dict__
        else:
            d = {k: getattr(conf, k) for k in fields}
    return repr_hash(d)


def repr_hash(obj):
    """Return a hash value of obj based on its repr()"""
    import hashlib
    return hashlib.md5(bytes(repr(obj), encoding='utf8')).hexdigest()


def to_path(maybe_str):
    if isinstance(maybe_str, str):
        return Path(maybe_str)
    return maybe_str


def json_load(json_file):
    """Load object from json file."""
    with to_path(json_file).open(encoding="utf8") as f:
        return json.load(f)


def json_dump(obj, json_file, **kwargs):
    """Dump obj to json file."""
    with to_path(json_file).open("w", encoding="utf8") as out:
        json.dump(obj, out, **kwargs)
        out.write("\n")


def json_dump_pandas(df, outfile, roundtrip_check=True, log=None, index=False):
    """Dump Pandas dataframe `df` to `outfile` in JSON format."""
    if index:
        raise NotImplementedError('TODO')
    df = df.reset_index()
    df_json = df.to_json(orient='table', index=False)

    if roundtrip_check:
        import pandas as pd
        from pandas.testing import assert_frame_equal
        assert_frame_equal(df, pd.read_json(df_json, orient='table'))

    with outfile.open('w') as out:
        out.write(df_json)

    if log is not None:
        log(outfile)


def jsonlines_load(jsonlines_file, max=None, skip=None, filter_fn=None):
    """Load objects from json lines file, i.e. a file with one
    serialized object per line."""
    if filter_fn is not None:
        yielded = 0
        for line in lines(jsonlines_file, skip=skip):
            obj = json.loads(line)
            if filter_fn(obj):
                yield obj
                yielded += 1
                if max and yielded >= max:
                    break
    else:
        yield from map(json.loads, lines(jsonlines_file, max=max, skip=skip))


def lines(file, max=None, skip=0, apply_func=str.strip, encoding="utf8"):
    """Iterate over lines in (text) file. Optionally skip first `skip`
    lines, only read the first `max` lines, and apply `apply_func` to
    each line. By default lines are stripped, set `apply_func` to None
    to disable this."""
    from itertools import islice
    if apply_func:
        with open(str(file), encoding=encoding) as f:
            for line in islice(f, skip, max):
                yield apply_func(line)
    else:
        with open(str(file), encoding=encoding) as f:
            for line in islice(f, skip, max):
                yield line


def ensure_serializable(_dict):
    """Converts non-serializable values in _dict to string.
    Main use case is to handle Path objects, which are not JSON-serializable.
    """
    def maybe_to_str(v):
        try:
            json.dumps(v)
        except TypeError:
            return str(v)
        return v
    return {k: maybe_to_str(v) for k, v in _dict.items()}


def mkdir(dir, parents=True, exist_ok=True):
    """Convenience function for Path.mkdir"""
    dir = to_path(dir)
    dir.mkdir(parents=parents, exist_ok=exist_ok)
    return dir


def clean_filename(s):
    for c in '/ ()':
        s = s.replace(c, '_')
    return s


# https://stackoverflow.com/a/1055378
def is_non_string_iterable(arg):
    """Return True if arg is an iterable, but not a string."""
    return (
        isinstance(arg, Iterable)
        and not isinstance(arg, six.string_types)
    )


def concat_dict_values(dicts):
    """Aggrgates multiple dictionaries into a single dictionary by
    concatenating the corresponding values of each dictionary
    """
    d = {}
    for _dict in dicts:
        for k, v in _dict.items():
            d.setdefault(k, []).append(v)
    return d


def get_formatter(fmt=None, datefmt=None):
    if not fmt:
        fmt = '%(asctime)s| %(message)s'
    if not datefmt:
        datefmt = "%Y-%m-%d %H:%M:%S"
    return logging.Formatter(fmt, datefmt=datefmt)


def get_logger(file=None, fmt=None, datefmt=None):
    log = logging.getLogger(__name__)
    formatter = get_formatter(fmt, datefmt)
    if not logging.root.handlers:
        logging.root.addHandler(logging.StreamHandler())
    logging.root.handlers[0].formatter = formatter
    if file:
        add_log_filehandler(log, file)
    return log


def add_log_filehandler(log, file):
    fhandler = logging.FileHandler(file)
    log.addHandler(fhandler)


class WithLog:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger()
        self.logger.setLevel(logging.INFO)
        self.log = self.logger.info


# source: https://stackoverflow.com/questions/16259923/how-can-i-escape-latex-special-characters-inside-django-templates  # NOQA
def tex_escape(text):
    """
        :param text: a plain text message
        :return: the message escaped to appear correctly in LaTeX
    """
    conv = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\^{}',
        '\\': r'\textbackslash{}',
        '<': r'\textless{}',
        '>': r'\textgreater{}',
    }
    regex = re.compile('|'.join(
        re.escape(key)
        for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


def df_to_latex(
        df,
        col_aligns="",
        header="",
        caption="",
        bold_max_cols=None,
        bold_max_row_idxs=None,
        na_rep="-",
        supertabular=False,
        midrule_after=None,
        colname_escape=True,
        ):
    """Converts a pandas dataframe into a latex table.

        :col_aligns: Optional format string for column alignments in the
                     tabular environment, e.g. 'l|c|rrr'. Will default
                     to right-align for numeric columns and left-align
                     for everthing else.
        :header: Optional header row. Default is the df column names.
        :bold_max_cols: Names of the columns among which the maximum
                        value will be bolded.
        :bold_max_row_idxs: Indexes of the rows in which maximum values
                        will be bolded.
        :na_rep: string representation of missing values.
        :supertabular: Whether to use the supertabular package (for long
                       tables).
        :midrule_after: Indexes of the rows after which a midrule should
                        be inserted.
        :colname_escape: Whether special latex characters in the column names
        should be escaped or not.
    """
    import numpy as np
    if bold_max_cols:
        max_cols_names = set(bold_max_cols)
    else:
        max_cols_names = set()
    if midrule_after is not None:
        if not hasattr(midrule_after, "__contains__"):
            midrule_after = set(midrule_after)

    def row_to_strings(row, i=None):
        is_bold_max_row = bold_max_row_idxs is None or i in bold_max_row_idxs
        if is_bold_max_row and bold_max_cols is not None:
            max_val = row[bold_max_cols].max()
        else:
            max_val = np.NaN

        def fmt(key, val):
            if key in max_cols_names and val == max_val:
                return r"\textbf{" + str(val) + "}"
            elif not isinstance(val, str) and np.isnan(val):
                return na_rep
            else:
                return str(val)

        return [fmt(key, val) for key, val in row.items()]

    if not col_aligns:
        dtype2align = {
            np.dtype("int"): "r",
            np.dtype("float"): "r"}
        col_aligns = "".join([
            dtype2align.get(df[colname].dtype, "l")
            for colname in df.columns])
    if header:
        if not header.endswith(r"\\"):
            header += r" \\"
    if not header:
        colnames = list(df.columns)
        if colname_escape:
            colnames = map(tex_escape, colnames)
        header = " & ".join(colnames) + r"\\"
    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rows.append(" & ".join(row_to_strings(row, i)) + "\\\\\n")
        if midrule_after and i in midrule_after:
            rows.append("\\midrule\n")
    if supertabular:

        latex_tbl = (r"""\tablehead{
\toprule
""" + header + "\n" + r"""\midrule
}
\tabletail{\bottomrule}
\bottomcaption{""" + caption + r"""}
\begin{supertabular}{""" + col_aligns + r"""}
""" + "".join(rows) + r"""\end{supertabular}
""")

    else:
        assert not caption
        latex_tbl = (r"\begin{tabular}{" + col_aligns + r"""}
\toprule
""" + header + r"""
\midrule
""" + "".join(rows) + r"""\bottomrule
\end{tabular}
""")

    return latex_tbl


class SubclassRegistry:
    '''Mixin that automatically registers all subclasses of the
    given class. Registered subclasses of superclass Class can be
    looked up via their names:
    >>> Class.get('subclass_name')

    The purpose of a SubclassRegistry is to make code like this
    uncessesary:
    >>> class Dataset_A():
            pass
    >>> class Dataset_B():
            pass
    >>> name2cls = {'A': Dataset_A, 'B': Dataset_B}
    >>> dataset = name2cls[dataset_name]()

    or code like

    >>> dataset = globals()[dataset_name]()

    With a SubclassRegistry, one can write something like this:

    >>> class Dataset(SubclassRegistry):
            pass
    >>> class A(Dataset):
            pass
    >>> class B(Dataset):
            pass
    >>> dataset = Dataset.get(dataset_name)()
    '''
    registered_classes = dict()
    registered_subclasses = defaultdict(set)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.registered_classes[cls.__name__.lower()] = cls
        for super_cls in cls.__mro__:
            if super_cls == cls:
                continue
            SubclassRegistry.registered_subclasses[super_cls].add(cls)

    @staticmethod
    def get(cls_name):
        return SubclassRegistry.registered_classes[cls_name]

    @classmethod
    def get_subclasses(cls):
        return SubclassRegistry.registered_subclasses.get(cls, {})

    @classmethod
    def get_subclass_names(cls, snakecase=False):
        names = [subcls.__name__ for subcls in cls.get_subclasses()]
        if snakecase:
            import stringcase
            names = list(map(stringcase.snakecase, names))
        return names


def take_singleton(items):
    """Returns the first item in `items` if `items` contains exactly
    one item, otherwise raises an exception.
    """
    exactly_one_taken = False
    for item in items:
        if exactly_one_taken:
            exactly_one_taken = False
            break
        exactly_one_taken = True
    if not exactly_one_taken:
        raise ValueError('items does not contain exactly one item')
    return item


def get_palette(categories, cmap=None):
    from bokeh.palettes import (
        Category20,
        Category20b,
        Category20c,
        viridis,
        )
    n_cat = len(set(categories))
    if cmap is not None:
        return cmap[n_cat]
    if n_cat <= 20:
        if n_cat <= 2:
            palette = Category20[3]
            return [palette[0], palette[-1]]
        else:
            return Category20[n_cat]
    if n_cat <= 40:
        return Category20[20] + Category20b[20]
    if n_cat <= 60:
        return Category20[20] + Category20b[20] + Category20c[20]
    return viridis(n_cat)


class BokehFigure:
    """
    data: Pandas dataframe or column dictionary
    plot_width: width of the bokeh plot
    plot_height: height of the bokeh plot
    reuse_figure: add plot to this figure instead of creating a new one
    sizing_mode: layout parameter
        see: https://docs.bokeh.org/en/latest/docs/reference/layouts.html#layout
    aspect_ratio: layout parameter
        see: https://docs.bokeh.org/en/latest/docs/reference/models/layouts.html#bokeh.models.LayoutDOM
    width: the width of the figure in pixels
    height: the height of the figure in pixels
    figure_kwargs: additional arguments that will be passed to
    bokeh.plotting.figure
    title: optional title of the plot
    colorbar: specifies whether to add a colorbar or not
    colorbar_title: optional colorbar title text
    colorticks: supply a list of tick values or set to False to disable
    labels: Optional text labels for each data point (or the name of a column
    if `data` is supplied)
    color: Optional color index for each datapoint, according to which it
    will be assigned a color (or the name of a column if `data` is supplied)
    raw_color: set this to true if `color` contains raw (RGB) color values
    instead of color indexes
    cmap_reverse: set to True to reverse the chosen colormap
    classes:  Optional class for each datapoint, according to which it will
    be colored in the plot (or the name of a column if `data` is supplied).
    cmap: name of a bokeh/colorcet colormap or a dictionary from values to
    color names
    tooltip_fields: column names for `data` or a column dictionary with data
    for each data point that will be used to add tooltip information
    plot_kwargs: additional arguments that will be pass to the bokeh glyph
    """
    def __init__(
            self,
            *,
            data=None,
            plot_width=None,
            plot_height=None,
            reuse_figure=None,
            sizing_mode='stretch_both',
            aspect_ratio=None,
            match_aspect=False,
            width=800,
            height=800,
            figure_kwargs=None,
            title=None,
            xaxis_label=None,
            yaxis_label=None,
            colorbar=False,
            colorbar_title=None,
            colorbar_ticks=True,
            labels=None,
            color=None,
            color_category=None,
            color_order=None,
            raw_colors=None,
            cmap=None,
            cmap_reverse=False,
            cmap_eq_hist=False,
            classes=None,
            class_category=None,
            tooltip_fields=None,
            **plot_kwargs,
            ):

        self.data = data

        def maybe_data_column(key_or_values):
            if isinstance(key_or_values, str) and self.data is not None:
                key = key_or_values
                values = self.data[key]
            else:
                values = key_or_values
            return values

        self.classes = classes
        self.class_category = class_category
        self.color = maybe_data_column(color)
        self.color_category = color_category
        self.color_order = color_order
        self.raw_colors = raw_colors
        self._cmap = cmap
        self.cmap_reverse = cmap_reverse
        self.cmap_eq_hist = cmap_eq_hist
        self.plot_kwargs = plot_kwargs
        self.tooltip_fields = tooltip_fields
        figure_kwargs = figure_kwargs or {}

        self.labels = maybe_data_column(labels)
        self.make_figure(
            plot_width=plot_width,
            plot_height=plot_height,
            reuse_figure=reuse_figure,
            sizing_mode=sizing_mode,
            aspect_ratio=aspect_ratio,
            match_aspect=match_aspect,
            width=width,
            height=height,
            **figure_kwargs,
            )
        self.add_title(title)
        self.add_axis_labels(xaxis_label=xaxis_label, yaxis_label=yaxis_label)
        if 'tooltips' not in figure_kwargs:
            self.add_tooltips()
        self.add_colorbar(
            colorbar=colorbar,
            colorbar_ticks=colorbar_ticks,
            title=colorbar_title,
            )

    def set_labels(self, labels):
        if isinstance(labels, str) and self.data is not None:
            self.labels = self.data[labels]
        else:
            self.labels = labels

    def make_figure(
            self,
            plot_width=None,
            plot_height=None,
            reuse_figure=None,
            sizing_mode='stretch_both',
            aspect_ratio=None,
            match_aspect=False,
            width=800,
            height=800,
            **figure_kwargs,
            ):
        from bokeh.plotting import figure
        if plot_width is not None:
            figure_kwargs['plot_width'] = plot_width
        if plot_height is not None:
            figure_kwargs['plot_height'] = plot_height
        if reuse_figure is None:
            self.figure = figure(
                tools=self.tools_str,
                sizing_mode=sizing_mode,
                aspect_ratio=aspect_ratio,
                match_aspect=match_aspect,
                width=width,
                height=height,
                **figure_kwargs,
                )
        else:
            self.figure = reuse_figure

    def add_title(self, title):
        if title:
            self.figure.title.text = title

    def add_axis_labels(self, xaxis_label=None, yaxis_label=None):
        self.figure.xaxis.axis_label = xaxis_label
        self.figure.yaxis.axis_label = yaxis_label

    @property
    def hover_tool_names(self):
        return None

    def add_tooltips(self):
        from bokeh.models import HoverTool
        hover = self.figure.select(dict(type=HoverTool))
        hover_entries = []
        if self.labels is not None:
            hover_entries.append(("label", "@label{safe}"))
        hover_entries.append(("(x, y)", "(@x, @y)"))
        if self.color is not None and self.color_category:
            hover_entries.append((self.color_category, "@color"))
        if self.classes is not None and self.class_category:
            hover_entries.append((self.class_category, "@class"))
        if self.tooltip_fields:
            for field in self.tooltip_fields:
                hover_entries.append((field, "@{" + field + '}'))
        hover.tooltips = dict(hover_entries)

    def add_colorbar(
            self,
            colorbar=False,
            colorbar_ticks=None,
            title=None
            ):
        if colorbar:
            from bokeh.models import ColorBar, BasicTicker, FixedTicker
            if colorbar_ticks:
                if colorbar_ticks is True:
                    ticker = BasicTicker()
                else:
                    ticker = FixedTicker(ticks=colorbar_ticks)
                ticker_dict = dict(ticker=ticker)
            else:
                ticker_dict = {}
            colorbar = ColorBar(
                color_mapper=self.color_mapper,
                title=title,
                height=240,
                **ticker_dict,
                )
            self.figure.add_layout(colorbar, 'right')

    @cached_property
    def tools(self):
        return [
            'crosshair',
            'pan',
            'wheel_zoom',
            'box_zoom',
            'reset',
            'hover',
            ]

    @cached_property
    def tools_str(self):
        return ','.join(self.tools)

    def save_or_show(self, outfile=None):
        if outfile:
            self.save(outfile=outfile)
        else:
            self.show()

    def save(self, outfile=None, write_png=False):
        from bokeh.plotting import output_file, save
        output_file(outfile)
        save(self.figure)
        if write_png:
            from bokeh.io import export_png
            png_file = outfile.with_suffix('.png')
            export_png(self.figure, filename=png_file)

    def show(self):
        from bokeh.plotting import show
        show(self.figure)

    @cached_property
    def color_conf(self):
        if self.raw_colors is not None:
            assert self.color is None
            if any(isinstance(c, str) for c in self.raw_colors):
                assert all(isinstance(c, str) for c in self.raw_colors)
            else:
                assert all(len(c) == 3 for c in self.raw_colors)
                assert self.cmap is None
            color_conf = {"field": "color"}
        elif self.color is not None:
            color_conf = {
                "field": "color",
                "transform": self.color_mapper}
        else:
            color_conf = "red"
        return color_conf

    @cached_property
    def source(self):
        from bokeh.models import ColumnDataSource
        source_dict = self.source_dict
        if self.labels is not None:
            source_dict["label"] = self.labels

        if self.raw_colors is not None:
            assert self.color is None
            if any(isinstance(c, str) for c in self.raw_colors):
                assert all(isinstance(c, str) for c in self.raw_colors)
            else:
                assert all(len(c) == 3 for c in self.raw_colors)
                assert self.cmap is None
            from bokeh.colors import RGB
            raw_colors = [RGB(*c) for c in self.raw_colors]
            source_dict["color"] = raw_colors
        elif self.color is not None:
            source_dict["color"] = self.color

        if self.classes is not None:
            source_dict["class"] = self.classes
        if self.tooltip_fields:
            if hasattr(self.tooltip_fields, 'items'):
                for k, v in self.tooltip_fields.items():
                    source_dict[k] = v
            elif self.data is not None:
                for f in self.tooltip_fields:
                    source_dict[f] = self.data[f]
        other_fields = ['factor_marker_col', 'factor_color_col']
        for field in other_fields:
            f = getattr(self, field, None)
            if f is not None:
                source_dict[f] = self.data[f]
        if self.data is not None:
            for col in self.data.columns:
                if col not in source_dict:
                    source_dict[col] = self.data[col]
        source = ColumnDataSource(source_dict)
        return source

    @cached_property
    def source_dict(self):
        raise NotImplementedError()

    def plot(self):
        self._plot()
        return self

    def _plot(self):
        raise NotImplementedError()

    @cached_property
    def color_mapper(self):
        from bokeh.models import (
            CategoricalColorMapper,
            LinearColorMapper,
            EqHistColorMapper,
            )
        if self.color is not None and any(isinstance(c, str) for c in self.color):
            assert all(isinstance(c, str) for c in self.color)
            palette = get_palette(self.color, cmap=self.cmap)
            color_order = self.color_order or sorted(set(self.color))
            return CategoricalColorMapper(
                    factors=color_order,
                    palette=palette)
        else:
            cmap = self.cmap
            if cmap is None:
                cmap = self.cmap_default
            elif isinstance(self.cmap, dict):
                cmap = self.cmap[max(self.cmap.keys())]
            if self.cmap_eq_hist:
                return EqHistColorMapper(cmap)
            return LinearColorMapper(cmap)

    @cached_property
    def cmap_default(self):
        from bokeh.palettes import Viridis256
        return Viridis256

    @cached_property
    def cmap(self):
        cmap = self._cmap
        cmap_reverse = self.cmap_reverse
        if cmap is not None:
            if isinstance(cmap, str):
                import bokeh.palettes
                # matplotib suffix for reverse color maps
                if cmap.endswith("_r"):
                    cmap_reverse = True
                    cmap = cmap[:-2]
                palette = getattr(bokeh.palettes, cmap, None)
                if palette is None:
                    import colorcet
                    palette = colorcet.palette[cmap]
                assert palette
                cmap = palette
            elif isinstance(cmap, dict):
                cmap = cmap[max(cmap.keys())]
            if cmap_reverse:
                if isinstance(cmap, dict):
                    new_cmap = {}
                    for k, v in cmap.items():
                        v = list(v)
                        v.reverse()
                        new_cmap[k] = v
                    cmap = new_cmap
                else:
                    cmap = list(cmap)
                    cmap.reverse()
        return cmap


class ScatterBokeh(BokehFigure):
    """Creates an interactive bokeh scatter plot.

    x: sequence of values or a column name if `data` is given
    y: sequence of values or a column name if `data` is given
    scatter_labels: set to True to scatter labels, i.e.,
    draw text instead of points
    data: Pandas dataframe or column dictionary
    marker: The name of a bokeh marker method, such as 'circle' or 'cross'
    """

    def __init__(
            self,
            *,
            x,
            y,
            scatter_labels=False,
            data=None,
            marker='circle',
            factor_marker_col=None,
            factor_marker_map=None,
            factor_color_col=None,
            factor_color_map=None,
            **kwargs,
            ):
        super().__init__(data=data, **kwargs)
        if data is not None:
            x = data[x]
            y = data[y]
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.scatter_labels = scatter_labels
        self.marker = marker
        self.factor_marker_col = factor_marker_col
        self.factor_marker_map = factor_marker_map
        self.factor_color_col = factor_color_col
        self.factor_color_map = factor_color_map

    @cached_property
    def source_dict(self):
        return dict(x=self.x, y=self.y)

    @property
    def glyph_name(self):
        return 'scatter'

    @property
    def hover_tool_names(self):
        return [self.glyph_name]

    def _plot(self):
        if self.scatter_labels:
            assert self.labels is not None
            from bokeh.models import Text
            glyph = Text(
                x="x",
                y="y",
                text="label",
                angle=0.0,
                text_color=self.color_conf,
                text_alpha=0.95,
                text_font_size="8pt",
                name=self.glyph_name,
                **self.plot_kwargs,
                )
            self.figure.add_glyph(self.source, glyph)
        else:
            plot_kwargs = dict(
                x='x',
                y='y',
                source=self.source,
                color=self.color_conf,
                name=self.glyph_name,
                **self.plot_kwargs,
                )
            if self.classes is not None:
                legend_field = 'class'
            else:
                legend_field = self.plot_kwargs.get('legend_field', None)
            if legend_field:
                plot_kwargs['legend_field'] = legend_field
                # sort by color field (default) or color_order (if provided)
                if self.color_order:
                    def key(series):
                        color2idx = {c: idx for idx, c in enumerate(self.color_order)}
                        return series.map(color2idx.__getitem__)
                else:
                    key = None
                sorted_source = self.source.to_df().sort_values(legend_field, key=key)
                plot_kwargs['source'] = self.source.from_df(sorted_source)

        if self.factor_marker_map:
            from bokeh.transform import factor_mark
            factors, markers = list(zip(*self.factor_marker_map.items()))
            marker = factor_mark(self.factor_marker_col, markers, factors)
        else:
            marker = None
        if self.factor_color_map:
            from bokeh.transform import factor_cmap
            factors, colors = list(zip(*self.factor_color_map.items()))
            color = factor_cmap(self.factor_color_col, colors, factors)
            plot_kwargs['color'] = color
        if marker is not None:
            self.figure.scatter(marker=marker, **plot_kwargs)
        elif self.marker is not None:
            marker_fn = getattr(self.figure, self.marker)
            marker_fn(**plot_kwargs)


def plot_embeddings_bokeh(
        emb,
        emb_method='UMAP',
        outfile=None,
        **kwargs,
        ):
    """
    Creates an interactive scatterplot of the embeddings contained in emb,
    using the bokeh library.
    emb: an array with dim (n_embeddings x 2) or (n_embeddings x emb_dim).
    In the latter case emb_method will be applied to project from emb_dim
    to dim=2.
    emb_method: "UMAP", "TSNE", or any other algorithm in sklearn.manifold
    outfile: If provided, save plot to this file instead of showing it
    kwargs: arguments passed to the ScatterBokeh plot
    """
    assert emb.shape[1] >= 2
    if emb.shape[1] > 2:
        from .embeddingutil import embed_2d
        emb = embed_2d(emb, emb_method)

    x, y = emb.T
    ScatterBokeh(x=x, y=y, **kwargs).plot().save_or_show(outfile=outfile)
