import numpy as np
from numpy.linalg import qr
from numpy.typing import ArrayLike


def complex_randn(*shape) -> ArrayLike:
    """
    Generate an array of complex numbers with random real and imaginary parts.

    Parameters:
        shape (tuple): The shape of the output array.

    Returns:
        (ArrayLike): An array of complex numbers with the specified shape.
    """
    return (np.random.randn(*shape).astype(np.float32)
            + 1j * np.random.randn(*shape).astype(np.float32))


def sample_haar_unitaries(n_samples: int, dim: int) -> ArrayLike:
    """
    Generate a set of Haar-random unitary matrices.

    Parameters:
        n_samples (int): The number of unitary matrices to generate.
        dim (int): The dimension of the unitary matrices.

    Returns:
        result (ArrayLike): An array of Haar-random unitary matrices of shape (n_samples, dim, dim).

    References:
        https://pennylane.ai/qml/demos/tutorial_haar_measure/
    """

    Zs = complex_randn(n_samples, dim, dim)
    result = np.empty((n_samples, dim, dim), dtype=np.complex64)

    for n, Z in enumerate(Zs):
        Q, R = qr(Z)
        lambd = np.diag(R)
        result[n] = Q @ np.diag(lambd) / np.abs(lambd)

    return result


def sample_haar_vectors(n_samples: int, dim: int) -> ArrayLike:
    """
    Generate random Haar vectors.

    Args:
        n_samples (int): Number of Haar vectors to generate.
        dim (int): Dimension of the Haar vectors.

    Returns:
        ArrayLike: Array of random Haar vectors.

    References:
        https://pennylane.ai/qml/demos/tutorial_haar_measure/
    """

    Zs = complex_randn(n_samples, dim, dim)
    result = np.empty((n_samples, dim), dtype=np.complex64)

    for n, Z in enumerate(Zs):
        Q, R = qr(Z)
        lambd = np.diag(R)
        result[n, :] = (Q @ np.diag(lambd) / np.abs(lambd))[0, :]

    return result


def sample_simplex_points(n_samples: int, dim: int) -> ArrayLike:
    """
    Generate random points on a simplex.

    Parameters:
        n_samples (int): Number of samples to generate.
        dim (int): Dimension of the simplex.

    Returns:
        ArrayLike: Array of shape (n_samples, dim) containing the generated points.
    """
    xis = np.random.rand(n_samples, dim-1).astype(np.float32)
    lambd = np.empty((n_samples, dim), dtype=np.float32)

    for n in range(n_samples):
        for k, xi in enumerate(xis[n]):
            lambd[n, k] = (1-xi**(1/(dim-k-1))) * (1-np.sum(lambd[n, :k]))
        lambd[n, -1] = 1 - np.sum(lambd[n, :-1])
    return lambd


def combine(probabilities: ArrayLike, unitaries: ArrayLike) -> ArrayLike:
    """
    Combines probabilities and unitaries to calculate density matrices.

    Args:
        probabilities (ArrayLike): Array of shape (n, d) representing the probabilities for each unitary.
        unitaries (ArrayLike): Array of shape (n, d, d) representing the unitary matrices.

    Returns:
        ArrayLike: Array of shape (n, d, d) representing the density matrices.

    Raises:
        AssertionError: If the number of probabilities and unitaries or their dimensions do not match.
    """

    assert probabilities.shape[0] == unitaries.shape[0], "The number of probabilities and unitaries must be the same."
    assert probabilities.shape[1] == unitaries.shape[1], "The dimension of the probabilities and unitaries must be the same."

    rhos = np.empty_like(unitaries)
    for n, (p, U) in enumerate(zip(probabilities, unitaries)):
        np.matmul(U, np.diag(p), out=rhos[n])
        np.matmul(rhos[n], U.conj().T, out=rhos[n])

    return rhos


def sample_from_ginibre_ensemble(n_samples: int, dim: int) -> ArrayLike:
    r"""
    Samples density matrices \(\rho = X^\dagger X / \text{tr} X^\dagger X\) where \(X\) is distributed according to the Ginibre ensemble.
    The Ginibre ensemble is the set of matrices with complex Gaussian entries with mean 0 and whose real and imaginary parts have unit variance.

    Parameters:
        n_samples (int): The number of samples to generate.
        dim (int): The dimension of the samples.

    Returns:
        ArrayLike: An array of density matrices.
    """

    X = complex_randn(n_samples, dim, dim)
    rhos = np.einsum('nlj,nlk->njk', X.conj(), X)

    for rho in rhos:
        rho /= np.trace(rho)

    return rhos


def sample_density_matrices(n_samples: int, dim: int, method: str = 'natural') -> ArrayLike:
    """
    Sample density matrices using different methods.

    Args:
        n_samples (int): The number of density matrices to sample.
        dim (int): The dimension of the density matrices.
        method (str, optional): The method to use for sampling. Defaults to 'natural'.

    Returns:
        ArrayLike: An array of sampled density matrices.

    Raises:
        AssertionError: If an unknown method is provided. Known methods are 'natural' and 'ginibre'.

    References:
        1. Życzkowski, K., Horodecki, P., Sanpera, A. & Lewenstein, M. Volume of the set of separable states. Phys. Rev. A 58, 883-892 (1998).


    """
    assert method in [
        'natural', 'ginibre'], f"Unknown method {method}. Choose from 'natural' or 'ginibre'."

    if method == 'natural':
        probabilities = sample_simplex_points(n_samples, dim)
        unitaries = sample_haar_unitaries(n_samples, dim)
        rhos = combine(probabilities, unitaries)
    elif method == 'ginibre':
        rhos = sample_from_ginibre_ensemble(n_samples, dim)

    return rhos
