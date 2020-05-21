# pyRPDE

A full-python implementation of the Recurrence Period Density Entropy metric.
It's based on the algorithm described in [^1], and on Max Little's 
R (and C) implementation in [^2]. It relies on Numba to make the return distance computation
as fast as possible without having to resort to Cython or C/C++ bindings.

## Installation

This package needs python >= 3.6, and relies on Numba, Scipy, and Numpy.
It's available on pypi, so a simple:

```shell
pip install pyrpde
```

should do the trick.

## Usage

There pretty much is only one function in this package that you should use, `rpde()`.
Here are its arguments:

```
    Arguments:
    ----------
    time_series: np.ndarray
        The input time series. Has to be float32, normalized to [-1,1]
    dim: int
        The dimension of the time series embeddings. 
        Defaults to 4
    tau: int
        The "stride" between each of the embedding points in a time series'
        embedding vector. Should be adjusted depending on the
        sampling rate of your input data.
        Defaults to 35.
    epsilon: float
        The size of the unit ball described in the RPDE algorithm.
        Defaults to 0.12.
    tmax: int, optional
        Maximum return distance (n1-n0), return distances higher than this
        are ignored. If set, can greatly improve the speed of the distance
        histogram computation (especially if your input time series has a lot of points).
        Defaults to None.
    parallel: boolean, optional
        Use the parallelized Numba implementation. The parallelization overhead
        might make this slower in certain situations. 
        Defaults to True.
    
    Returns
    -------
    rpde: float
        Value of the RPDE
    histogram: np.ndarray
        1-dimensional array corresponding to the histogram of the return
        distances
```

**NOTE**: the default values for `tau`, `dim` and `epsilon` are adapted from [^1] and [^2],
 to work on 22.5Khz PCM audio. You should probably use `tau=25` for 16Khz and `tau=50`
 for 48KHz audio. 

Here's an example: 

```python
from pyrpde import  rpde
from scipy.io.wavfile import read

# make sure your audio data is in float32. Else, either use librosa or 
# normalize it to [-1,1] by dividing it by 2 ** 16 if it's 16bit PCM
rate, data = read("audio_data.wav")
entropy, histogram = rpde(data, tau=30, dim=4, epsilon=0.01, tmax=1500)

```

[^1]: http://www.biomedical-engineering-online.com/content/6/1/23
[^2]: http://www.maxlittle.net/software/index.php
