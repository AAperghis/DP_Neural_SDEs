import numpy as np


def resample_from_base(
    x_base: np.ndarray, dt_base: float, dt_target: float, T: float
) -> np.ndarray:
    """
    Sample the base OU at multiples of dt_target up to T.
    Assumes dt_target is an integer multiple of dt_base (recommended).
    """
    n_target = int(np.round(T / dt_target))
    step = int(np.round(dt_target / dt_base))
    idx = np.arange(0, n_target + 1) * step
    return x_base[idx, :]


def ou_generate_uniform(
    N: int,
    dt: float,
    theta: float = 0.01,
    mu: np.ndarray = np.zeros(3),
    sigma: np.ndarray = np.zeros(3),
    x0: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Exact OU discretization on a uniform grid.
    SDE: dX = theta*(mu - X) dt + sigma dW
    Returns array of length N+1 (including initial value at t=0).
    """
    if rng is None:
        rng = np.random.default_rng()
    if x0 is None:
        # draw from stationary distribution for warm start
        x0 = mu + rng.normal(scale=sigma / np.sqrt(2.0 * theta), size=3)
    x = np.empty([N + 2, 3], dtype=float)
    x[0, :] = x0

    phi = np.exp(-theta * dt)  # AR coefficient
    q = sigma * np.sqrt((1.0 - phi**2) / (2.0 * theta))  # noise std

    z = rng.normal(size=N + 1)
    for n in range(N + 1):
        x[n + 1] = mu + (x[n] - mu) * phi + q * z[n]
    return x
