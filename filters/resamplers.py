import numpy as np

# A single module-level Generator is used as the default source of randomness
# so successive calls to the resamplers draw from the same stream (matching
# the behaviour of the legacy ``numpy.random`` global state). Callers who need
# reproducible output — tests, notebooks pinning a seed — should pass their
# own ``numpy.random.Generator`` via the ``rng`` argument.
_default_rng = np.random.default_rng()


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return _default_rng if rng is None else rng


def _normalised_weights(weights):
    """Return weights that sum to exactly one, falling back to uniform."""

    w = np.asarray(weights, dtype=float)
    total = w.sum()
    if total <= 0.0:
        return np.full(len(w), 1.0 / len(w))
    return w / total


def _cumulative(weights):
    """CDF of ``weights`` with the last bin forced to 1.0.

    Without the final clamp, floating-point drift (or weights that do not sum
    to one) leaves ``cumulative_sum[-1] < 1`` and ``searchsorted`` can return
    an index of ``N`` — one past the last particle.
    """

    c = np.cumsum(_normalised_weights(weights))
    c[-1] = 1.0
    return c


def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))


def residual_resample(weights, rng=None):
    """Performs the residual resampling algorithm used by particle filters.

    Based on observation that we don't need to use random numbers to select most of the weights. Take int(N*w^i) samples of each particle i, and then resample any remaining using a standard resampling algorithm.

    References
    ----------
    J. S. Liu and R. Chen. Sequential Monte Carlo methods for dynamic systems. Journal of the American Statistical Association, 93(443):1032–1044, 1998.
    """
    rng = _resolve_rng(rng)

    N = len(weights)
    weights = _normalised_weights(weights)
    indexes = np.zeros(N, "i")

    # take int(N*w) copies of each weight, which ensures particles with the
    # same weight are drawn uniformly
    num_copies = (np.floor(N * weights)).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]):  # make n copies
            indexes[k] = i
            k += 1

    # use multinormal resample on the residual to fill up the rest. This
    # maximizes the variance of the samples
    # ``num_copies`` counts particles while ``weights`` are probabilities;
    # the fractional part must be computed in the same (count) units.
    residual = np.maximum(N * weights - num_copies, 0.0)
    total = np.sum(residual)
    if total <= 0.0:
        # Every particle already received an integer number of copies, or the
        # weights do not sum to one. Dividing by zero here used to emit NaN and
        # index the particles with garbage; fall back to uniform.
        residual = np.full(N, 1.0 / N)
        total = 1.0
    residual = residual / total
    cumulative_sum = np.cumsum(residual)
    cumulative_sum[-1] = 1.0  # avoid round-off errors: ensures sum is exactly one
    indexes[k:N] = np.searchsorted(cumulative_sum, rng.random(N - k))

    return indexes


def stratified_resample(weights, rng=None):
    """Performs the stratified resampling algorithm used by particle filters.

    This algorithms aims to make selections relatively uniformly across the particles. It divides the cumulative sum of the weights into N equal divisions, and then selects one particle randomly from each division. This guarantees that each sample is between 0 and 2/N apart.
    """
    rng = _resolve_rng(rng)

    N = len(weights)
    # make N subdivisions, and choose a random position within each one
    positions = (rng.random(N) + np.arange(N)) / N

    return np.searchsorted(_cumulative(weights), positions)


def systematic_resample(weights, rng=None):
    """Performs the systemic resampling algorithm used by particle filters.

    This algorithm separates the sample space into N divisions. A single random offset is used to to choose where to sample from for all divisions. This guarantees that every sample is exactly 1/N apart.
    """
    rng = _resolve_rng(rng)
    N = len(weights)

    # make N subdivisions, and choose positions with a consistent random offset
    positions = (rng.random() + np.arange(N)) / N

    return np.searchsorted(_cumulative(weights), positions)


def multinomial_resample(weights, rng=None):
    """This is the naive form of roulette sampling where we compute the
    cumulative sum of the weights and then use binary search to select the resampled point based on a uniformly distributed random number. Run time is O(n log n). You do not want to use this algorithm in practice; for some reason it is popular in blogs and online courses so I included it for reference.
    """
    rng = _resolve_rng(rng)
    return np.searchsorted(_cumulative(weights), rng.random(len(weights)))
