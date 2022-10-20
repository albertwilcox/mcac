import numpy as np
import os


class ReplayBuffer:
    """
    This replay buffer uses numpy to efficiently store arbitrary data. Keys can be whatever,
    but once you push data to the buffer new data must all have the same keys (to keep parallel
    arrays parallel).
    """
    def __init__(self, size=100000):
        self.size = size

        self._reset()

    def _reset(self):
        self.data = {}
        self._index = 0
        self._len = 0

    def store_trajectory(self, transistions):
        """
        Stores transitions
        :param transistions: a list of dictionaries encoding transitions. Keys can be anything
        """
        assert transistions[-1]['done'] > 0, "Last transition must be end of trajectory"
        for transition in transistions:
            self.store_transition(transition)

    def store_transition(self, transition):
        if len(self.data) > 0:
            key_set = set(self.data)
        else:
            key_set = set(transition)

        assert key_set == set(transition), "Expected transition to have keys %s, got %s" \
                                           % (key_set, set(transition))

        for key in key_set:
            data = self.data.get(key, None)
            new_data = np.array(transition[key])
            if data is None:
                data = np.zeros((self.size, *new_data.shape), dtype=new_data.dtype)
            data[self._index] = new_data
            self.data[key] = data

        self._index = (self._index + 1) % self.size
        self._len = min(self._len + 1, self.size)

    def sample(self, batch_size, ensemble=0):
        if ensemble == 0:
            indices = np.random.randint(len(self), size=batch_size)
        elif ensemble > 0:
            indices = np.random.randint(len(self), size=(ensemble, batch_size))
        else:
            raise ValueError("ensemble size cannot be negative")

        return self._im_to_float({key: self.data[key][indices] for key in self.data})

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample_chunk(self, batch_size, length, ensemble=0):
        if ensemble == 0:
            idxs = np.asarray([self._sample_idx(length) for _ in range(batch_size)])
        elif ensemble > 0:
            idxs = np.asarray([[self._sample_idx(length) for _ in range(batch_size)]
                               for _ in range(ensemble)])
        else:
            raise ValueError("ensemble size cannot be negative")
        out_dict = {}
        for key in self.data:
            out = self.data[key][idxs]
            out_dict[key] = out

        # Everything after the end of a trajectory gets masked out
        if 'mask' in out_dict:
            mask = out_dict['mask']
            for i in range(1, mask.shape[1]):
                mask[:, i] = mask[:, i] * mask[:, i-1]
        return self._im_to_float(out_dict)

    def _im_to_float(self, out_dict):
        for key in out_dict:
            if key in ('obs', 'next_obs'):
                if out_dict[key].dtype == np.uint8:
                    out_dict[key] = out_dict[key] / 255
        return out_dict

    def _sample_idx(self, length):
        idx = np.random.randint(0, len(self) - length)
        idxs = np.arange(idx, idx + length) % self.size
        return idxs

    def save(self, folder):
        os.makedirs(folder)
        for key, item in self.data.items():
            np.save(os.path.join(folder, f'{key}.npy'), item[:self._len])

    def load(self, folder):
        self._reset()

        files = os.listdir(folder)
        for file in files:
            dat = np.load(os.path.join(folder, file))
            name = file.split('.')[0]
            self.data[name] = dat

        self._len = len(self.data[list(set(self.data))[0]])
        self._index = self._len

    def __len__(self):
        return self._len
