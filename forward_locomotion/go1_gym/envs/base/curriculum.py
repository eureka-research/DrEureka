import numpy as np
from matplotlib import pyplot as plt


def is_met(scale, l2_err, threshold):
    return (l2_err / scale) < threshold


def key_is_met(metric_cache, config, ep_len, target_key, env_id, threshold):
    # metric_cache[target_key][env_id] / ep_len
    scale = 1
    l2_err = 0
    return is_met(scale, l2_err, threshold)


class Curriculum:
    def set_to(self, low, high, value=1.0):
        inds = np.logical_and(
            self.grid >= low[:, None],
            self.grid <= high[:, None]
        ).all(axis=0)

        self.weights[inds] = value

    def __init__(self, seed, **key_ranges):
        self.rng = np.random.RandomState(seed)

        self.cfg = cfg = {}
        for key, v_range in key_ranges.items():
            cfg[key] = np.linspace(*v_range)

        self.bin_sizes = {key: arr[1] - arr[0] for key, arr in cfg.items()}

        self._raw_grid = np.stack(np.meshgrid(*cfg.values(), indexing='ij'))
        self.keys = [*key_ranges.keys()]
        self.grid = self._raw_grid.reshape([len(self.keys), -1])
        # self.grid = np.stack([params.flatten() for params in raw_grid])

        self._l = l = len(self.grid[0])
        self.ls = {key: len(self.cfg[key]) for key in self.cfg.keys()}

        self.weights = np.zeros(l)
        self.indices = np.arange(l)

    def __len__(self):
        return self._l

    def __getitem__(self, *keys):
        pass

    def update(self, **kwargs):
        # bump the envelop if
        pass

    def sample_bins(self, batch_size):
        """default to uniform"""
        inds = self.rng.choice(self.indices, batch_size, p=self.weights / self.weights.sum())
        # return np.stack([params[inds] for params in self.grid.values()]).T, inds
        return self.grid.T[inds], inds

    def sample_uniform_from_cell(self, centroids):
        bin_sizes = np.array([*self.bin_sizes.values()])
        low, high = centroids + bin_sizes / 2, centroids - bin_sizes / 2
        return self.rng.uniform(low, high)

    def sample(self, batch_size):
        cgf_centroid, inds = self.sample_bins(batch_size)
        return np.stack([self.sample_uniform_from_cell(v_range) for v_range in cgf_centroid]), inds


class SumCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.success = np.zeros(len(self))
        self.trials = np.zeros(len(self))

    def update(self, bin_inds, l1_error, threshold):
        is_success = l1_error < threshold
        self.success[bin_inds[is_success]] += 1
        self.trials[bin_inds] += 1

    def success_rates(self, *keys):
        s_rate = self.success / (self.trials + 1e-6)
        s_rate = s_rate.reshape(list(self.ls.values()))
        marginals = tuple(i for i, key in enumerate(self.keys) if key not in keys)
        if marginals:
            return s_rate.mean(axis=marginals)
        return s_rate


class RewardThresholdCurriculum(Curriculum):
    def __init__(self, seed, **kwargs):
        super().__init__(seed, **kwargs)

        self.episode_reward_lin = np.zeros(len(self))
        self.episode_reward_ang = np.zeros(len(self))
        self.episode_lin_vel_raw = np.zeros(len(self))
        self.episode_ang_vel_raw = np.zeros(len(self))
        self.episode_duration = np.zeros(len(self))

    def get_local_bins(self, bin_inds, range=0.1):
        adjacent_inds = np.logical_and(
            self.grid[:, None, :].repeat(len(bin_inds), axis=1) >= self.grid[:, bin_inds, None] - range,
            self.grid[:, None, :].repeat(len(bin_inds), axis=1) <= self.grid[:, bin_inds, None] + range
        ).all(axis=0)

        return adjacent_inds

    def update(self, bin_inds, lin_vel_rewards, ang_vel_rewards, lin_vel_threshold, ang_vel_threshold, local_range=0.5):
        self.episode_reward_lin[bin_inds] = lin_vel_rewards
        self.episode_reward_ang[bin_inds] = ang_vel_rewards

        is_success = ((lin_vel_rewards > lin_vel_threshold) * (ang_vel_rewards > ang_vel_threshold))
        self.weights[bin_inds[is_success]] = np.clip(self.weights[bin_inds[is_success]] + 0.2, 0, 1)
        adjacents = self.get_local_bins(bin_inds[is_success], range=local_range)
        for adjacent in adjacents:
            adjacent_inds = np.array(adjacent.nonzero()[0])
            self.weights[adjacent_inds] = np.clip(self.weights[adjacent_inds] + 0.2, 0, 1)

    def log(self, bin_inds, lin_vel_raw=None, ang_vel_raw=None, episode_duration=None):
        self.episode_lin_vel_raw[bin_inds] = lin_vel_raw.cpu().numpy()
        self.episode_ang_vel_raw[bin_inds] = ang_vel_raw.cpu().numpy()
        self.episode_duration[bin_inds] = episode_duration.cpu().numpy()


if __name__ == '__main__':
    r = RewardThresholdCurriculum(100, x=(-1, 1, 5), y=(-1, 1, 2), z=(-1, 1, 11))

    assert r._raw_grid.shape == (3, 5, 2, 11), "grid shape is wrong: {}".format(r.grid.shape)

    low, high = np.array([-1.0, -0.6, -1.0]), np.array([1.0, 0.6, 1.0])

    # r.set_to(low, high, value=1.0)

    adjacents = r.get_local_bins(np.array([10, ]), range=0.5)
    for adjacent in adjacents:
        adjacent_inds = np.array(adjacent.nonzero()[0])
        print(adjacent_inds)
        r.update(bin_inds=adjacent_inds, lin_vel_rewards=np.ones_like(adjacent_inds),
                 ang_vel_rewards=np.ones_like(adjacent_inds), lin_vel_threshold=0.0, ang_vel_threshold=0.0,
                 local_range=0.5)

    samples, bins = r.sample(10_000)

    plt.scatter(*samples.T[:2])
    plt.show()
