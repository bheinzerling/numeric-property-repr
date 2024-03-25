from torch.utils.data import (
    Subset,
    DataLoader,
    BatchSampler,
    RandomSampler,
    )

from util import (
    cached_property,
    )


class Loaders():
    split_names = ['train', 'dev', 'test']
    use_batch_sampler = False

    def loaders(
            self,
            batch_size,
            *args,
            eval_batch_size=None, split_names=None,
            use_batch_sampler=False,
            log=None,
            **kwargs):
        if not split_names:
            split_names = self.split_names
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size or batch_size
        self.use_batch_sampler = use_batch_sampler
        loaders = {
            split_name: getattr(
                self, split_name + '_loader')(*args, **kwargs)
            for split_name in split_names}
        loaders['train_inference'] = DataLoader(
            Subset(self.train, list(range(len(self.dev)))),
            batch_size=eval_batch_size)
        if log is not None:
            for split_name, loader in loaders.items():
                log(f'{split_name} batches: {len(loader)}')
        return loaders

    def train_loader(self, *args, **kwargs):
        assert 'train' in self.split_names
        batch_size = (
            kwargs.pop('batch_size')
            if 'batch_size' in kwargs
            else self.batch_size)
        if self.use_batch_sampler:
            batch_sampler = BatchSampler(
                RandomSampler(self.train), batch_size, drop_last=False)
            return DataLoader(
                self.train, *args, batch_sampler=batch_sampler, **kwargs)
        return DataLoader(
            self.train, *args, batch_size=batch_size, **kwargs)

    def dev_loader(self, *args, **kwargs):
        assert 'dev' in self.split_names
        batch_size = (
            kwargs.pop('batch_size')
            if 'batch_size' in kwargs
            else self.eval_batch_size)
        return DataLoader(
            self.dev, *args, batch_size=batch_size, **kwargs, shuffle=False)

    def test_loader(self, *args, **kwargs):
        assert 'test' in self.split_names
        batch_size = (
            kwargs.pop('batch_size')
            if 'batch_size' in kwargs
            else self.eval_batch_size)
        return DataLoader(
            self.test, *args, batch_size=batch_size, **kwargs, shuffle=False)


class FixedSplits(Loaders):
    def __init__(self, train=None, dev=None, test=None):
        self.train = train
        self.dev = dev
        self.test = test


class WithSplits:
    split_names = ['train', 'dev', 'test']

    def __init__(self, *args, do_check_overlap=True, **kwargs):
        self.do_check_overlap = do_check_overlap

    @cached_property
    def splits(self):
        raise NotImplementedError

    def load_raw_split(self, split_name):
        raise NotImplementedError

    @cached_property
    def train_raw(self):
        return self.load_raw_split('train')

    @cached_property
    def dev_raw(self):
        return self.load_raw_split('dev')

    @cached_property
    def test_raw(self):
        return self.load_raw_split('test')

    @cached_property
    def raw(self):
        split_name2split = {
            split_name: getattr(self, split_name + '_raw')
            for split_name in self.split_names
            }

        if hasattr(self, 'instance_id') and getattr(self, 'do_check_overlap', True):
            self.check_splits_overlap(split_name2split.values())

        return split_name2split

    @property
    def train_loader(self):
        return self.splits.train_loader(batch_size=self.conf.batch_size)

    @property
    def dev_loader(self):
        return self.splits.dev_loader(batch_size=self.conf.eval_batch_size)

    @property
    def test_loader(self):
        return self.splits.test_loader(batch_size=self.conf.eval_batch_size)

    def check_splits_overlap(self, splits):
        """Make sure that there isn't any overlap between the splits, i.e.,
        there shouldn't be any instances that are part of more than one split.
        """
        from itertools import combinations
        idss = [set(map(self.instance_id, split)) for split in splits]
        for ids0, ids1 in combinations(idss, 2):
            overlap = ids0 & ids1
            assert not overlap


class TensorDictDataset():
    """Like Pytorch's TensorDict, but instead of storing multiple tensors
    in a tuple, stores tensors in a dict."""
    def __init__(self, **tensors):
        assert all(
            next(iter(tensors.values())).size(0) == t.size(0)
            for t in tensors.values()
            )
        self.tensors = tensors

    def __getitem__(self, index):
        return {k: t[index] for k, t in self.tensors.items()}

    def __len__(self):
        return len(next(iter(self.tensors.values())))
