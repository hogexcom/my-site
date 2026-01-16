import numpy as np

DEFAULT_TRANSFORMS = [
    [0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.01],
    [0.85, 0.04, -0.04, 0.85, 0.0, 1.6, 0.85],
    [0.20, -0.26, 0.23, 0.22, 0.0, 1.6, 0.07],
    [-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, 0.07],
]

def barnsley_fern(n=200_000, burn_in=100, seed=None, transforms=None):
    rng = np.random.default_rng(seed)

    x = np.zeros(n, dtype=np.float32)
    y = np.zeros(n, dtype=np.float32)

    if transforms is None:
        transforms = DEFAULT_TRANSFORMS

    transforms = np.asarray(transforms, dtype=np.float32)
    if transforms.ndim != 2 or transforms.shape[1] < 7:
        raise ValueError("transforms must be an array of [a, b, c, d, e, f, p]")

    a = transforms[:, 0]
    b = transforms[:, 1]
    c = transforms[:, 2]
    d = transforms[:, 3]
    e = transforms[:, 4]
    f = transforms[:, 5]
    p = transforms[:, 6]
    p_sum = float(np.sum(p))
    if p_sum <= 0:
        p = np.full_like(p, 1.0 / len(p))
    else:
        p = p / p_sum
    cum_probs = np.cumsum(p)
    cum_probs[-1] = 1.0

    for _ in range(max(1, int(burn_in))):
        r = rng.random(n, dtype=np.float32)
        idx = np.searchsorted(cum_probs, r, side="right")

        x_new = np.empty_like(x)
        y_new = np.empty_like(y)

        for i in range(len(p)):
            mask = idx == i
            if not np.any(mask):
                continue
            x_new[mask] = a[i] * x[mask] + b[i] * y[mask] + e[i]
            y_new[mask] = c[i] * x[mask] + d[i] * y[mask] + f[i]

        x = x_new
        y = y_new

    return x.tolist(), y.tolist()
