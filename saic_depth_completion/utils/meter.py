from collections import deque

import torch


class Statistics(object):
    def __init__(self, maxlen=20):
        self.enum = deque(maxlen=maxlen)
        self.denum = deque(maxlen=maxlen)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.enum.clear()
        self.denum.clear()

    def update(self, value, n):
        self.enum.append(value)
        self.denum.append(n)
        self.count += n
        self.total += value

    @property
    def median(self):
        enum = torch.tensor(list(self.enum))
        denum = torch.tensor(list(self.denum))
        sequence = enum / denum
        return sequence.median().item()

    @property
    def avg(self):
        enum = torch.tensor(list(self.enum))
        denum = torch.tensor(list(self.denum))
        avg = enum.sum() / denum.sum()
        return  avg.item()

    @property
    def global_avg(self):
        return self.total / self.count


class Meter:
    def __init__(self, metric_fn, maxlen=20):
        self.metric_fn = metric_fn
        self.stats = Statistics(maxlen)

    def reset(self):
        self.stats.reset()

    def update(self, pred, gt):
        value = self.metric_fn(pred, gt)
        if isinstance(value, tuple):
            self.stats.update(value[0].cpu(), value[1])
        else:
            self.stats.update(value.item(), 1)

    @property
    def median(self):
        return self.stats.median
    @property
    def avg(self):
        return self.stats.avg

    @property
    def global_avg(self):
        return self.stats.global_avg

class AggregatedMeter(object):
    def __init__(self, metrics, maxlen=20, delimiter=' # '):
        self.delimiter = delimiter
        self.meters = {
            k: Meter(v, maxlen) for k, v in metrics.items()
        }

    def reset(self):
        for v in self.meters.values():
            v.reset()

    def update(self, pred, gt):
        for v in self.meters.values():
            v.update(pred, gt)

    @property
    def suffix(self):
        suffix = []
        for k, v in self.meters.items():
            suffix.append(
                "{}: {:.4f} ({:.4f})".format(k, v.median, v.global_avg)
            )
        return self.delimiter.join(suffix)