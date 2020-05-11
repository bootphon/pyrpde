import numpy as np
import numba as nb
from scipy.stats import entropy

# TODO : change to a scipy-style function that outputs the entropy and the recurrence histogram

class RPDE:
    def __init__(self,
                 dim: int,
                 tau: int,
                 epsilon: float,
                 tmax: Optional[int] = None,
                 parallel: bool = True):
        super().__init__(dim=dim, tau=tau, epsilon=epsilon, tmax=tmax)
        self.epsilon, self.tau, self.dim = epsilon, tau, dim
        self.tmax = tmax
        self.parallel = parallel

    @staticmethod
    @nb.jit(nb.int32[:](nb.float32[:, :], nb.float32, nb.int32), nopython=True)
    def recurrence_histogram(ts: np.ndarray, epsilon: float, t_max: int):
        """Numba implementation of the recurrence histogram described in
        http://www.biomedical-engineering-online.com/content/6/1/23

        :param ts: Time series of dimension (T,N)
        :param epsilon: Recurrence ball radius
        :param t_max: Maximum distance for return.
        Larger distances are not recorded. Set to -1 for infinite distance.
        """
        return_distances = np.zeros(len(ts), dtype=np.int32)
        for i in np.arange(len(ts)):
            # finding the first "out of ball" index
            first_out = len(ts)  # security
            for j in np.arange(i + 1, len(ts)):
                if 0 < t_max < j - i:
                    break
                d = np.linalg.norm(ts[i] - ts[j])
                if d > epsilon:
                    first_out = j
                    break

            # finding the first "back to the ball" index
            for j in np.arange(first_out + 1, len(ts)):
                if 0 < t_max < j - i:
                    break
                d = np.linalg.norm(ts[i] - ts[j])
                if d < epsilon:
                    return_distances[j - i] += 1
                    break
        return return_distances

    @staticmethod
    @nb.jit(nb.int32[:](nb.float32[:, :], nb.float32, nb.int32), parallel=True, nopython=True)
    def parallel_recurrence_histogram(ts: np.ndarray, epsilon: float,
                                      t_max: int):
        """Parallelized implem of the recurrence histogram. Works the same,
        but adapted to parallelized computing (automatically done by Numba)."""
        return_distances = np.zeros(len(ts), dtype=np.int32)

        # this is the parallized loop
        for i in nb.prange(len(ts)):
            # finding the first "out of ball" index
            first_out = len(ts)  # security
            for j in np.arange(i + 1, len(ts)):
                if t_max > 0 and j - i > t_max:
                    break
                d = np.linalg.norm(ts[i] - ts[j])
                if d > epsilon:
                    first_out = j
                    break

            # finding the first "back to the ball" index
            for j in np.arange(first_out + 1, len(ts)):
                if t_max > 0 and j - i > t_max:
                    break
                d = np.linalg.norm(ts[i] - ts[j])
                if d < epsilon:
                    return_distances[i] = j - i
                    break

        # building histogram, can't be parallel
        # (potential concurrent access to histogram items)
        distance_histogram = np.zeros(len(ts), dtype=np.int32)
        for i in np.arange(len(ts)):
            if return_distances[i] != 0:
                distance_histogram[return_distances[i]] += 1
        return distance_histogram

    @staticmethod
    def embed_time_series(data: np.ndarray, dim: int, tau: int):
        """Embeds the time series, tested against the pyunicorn implementation
        of that transform"""
        # TODO : check that this implem holds when the data shape isn't unidimentional
        embed_points_nb = data.shape[0] - (dim - 1) * tau
        # embed_mask is of shape (embed_pts, dim)
        embed_mask = np.arange(embed_points_nb)[:, np.newaxis] + np.arange(dim)
        tau_offsets = np.arange(dim) * (tau - 1)  # shape (dim,1)
        embed_mask = embed_mask + tau_offsets
        return data[embed_mask]

    def __call__(self, time_series: np.ndarray) -> float:
        # converting the sound array to the right format
        # (RPDE expects the array to be floats in [-1,1]
        # TODO: check DType
        if sample_data.array.dtype == np.int16:
            data = (sample_data.array / (2**16)).astype(np.float32)
        else:
            data = sample_data.array.astype(np.float32)

        #  building the time series, and computing the recurrence histogram
        embedded_ts = self.embed_time_series(data, self.dim, self.tau)
        if self.tmax is None:
            rec_histogram = self.parallel_recurrence_histogram(
                embedded_ts, self.epsilon, -1)
            # tmax is the highest non-zero value in the histogram
            tmax_idx = np.argwhere(rec_histogram != 0).flatten()[-1]
        else:
            rec_histogram = self.parallel_recurrence_histogram(
                embedded_ts, self.epsilon, self.tmax)
            tmax_idx = self.tmax
        # culling the histogram at the tmax index
        culled_histogram = rec_histogram[:tmax_idx]

        # normalizing the histogram (to make it a probability)
        # and computing its entropy
        if culled_histogram.sum():
            normalized_histogram = culled_histogram / culled_histogram.sum()
            histogram_entropy = entropy(normalized_histogram)
            return histogram_entropy / np.log(culled_histogram.size)
        else:
            return 0.0