import numpy as np


class Tracker:
    def __init__(
            self, subset, target, snapshoter, init_state=float("inf"), delay=10, compare_fn=np.less, eps=0.03
    ):
        self.subset = subset
        self.target = target
        self.snapshoter = snapshoter
        self.state = init_state
        self.delay = delay
        self.compare_fn = compare_fn
        self.epoch_counter = 0
        self.eps = eps

    def update(self, subset, metric_state):
        if subset != self.subset: return

        self.epoch_counter += 1

        if self.epoch_counter < self.delay: return

        # save best model
        if self.compare_fn(metric_state[self.target], self.state):
            self.state = metric_state[self.target]
            self.snapshoter.save("snapshot_{}_{:.4f}".format(self.target, self.state))
            return

        # save model from epsilon neighborhood
        if np.abs(self.state - metric_state[self.target]) < self.eps:
            self.snapshoter.save("snapshot_eps_{}_{:.4f}".format(self.target, self.state))


class ComposedTracker:
    def __init__(self, trackers):
        self.trackers = trackers
    def update(self, subset, metric_state):
        for tracker in self.trackers:
            tracker.update(subset, metric_state)