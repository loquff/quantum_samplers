import numpy as np
import quantum_samplers as qs


def test_complex_randn():
    # Test case 1: shape = (2, 2)
    shape = (2, 2)
    result = qs.complex_randn(*shape)
    assert result.shape == shape
    assert np.all(np.iscomplex(result))

    # Test case 2: shape = (3, 3, 3)
    shape = (3, 3, 3)
    result = qs.complex_randn(*shape)
    assert result.shape == shape
    assert np.all(np.iscomplex(result))

    # Test case 3: shape = (1,)
    shape = (1,)
    result = qs.complex_randn(*shape)
    assert result.shape == shape
    assert np.all(np.iscomplex(result))

    print("All test cases passed!")


test_complex_randn()


def test_sample_haar_unitaries():
    # Test case 1: n_samples = 2, dim = 2
    n_samples = 2
    dim = 2
    result = qs.sample_haar_unitaries(n_samples, dim)
    assert result.shape == (n_samples, dim, dim)
    for matrix in result:
        assert np.allclose(matrix @ matrix.conj().T, np.eye(dim), atol=1e-6)

    # Test case 2: n_samples = 3, dim = 3
    n_samples = 3
    dim = 3
    result = qs.sample_haar_unitaries(n_samples, dim)
    assert result.shape == (n_samples, dim, dim)
    for matrix in result:
        assert np.allclose(matrix @ matrix.conj().T, np.eye(dim), atol=1e-6)

    print("All test cases passed!")

    # Test case 2: n_samples = 4, dim = 4
    n_samples = 4
    dim = 4
    result = qs.sample_haar_unitaries(n_samples, dim)
    assert result.shape == (n_samples, dim, dim)
    for matrix in result:
        assert np.allclose(matrix @ matrix.conj().T, np.eye(dim), atol=1e-6)

    print("All test cases passed!")


test_sample_haar_unitaries()


def test_sample_haar_vectors():
    # Test case 1: n_samples = 2, dim = 2
    n_samples = 2
    dim = 2
    result = qs.sample_haar_vectors(n_samples, dim)
    assert result.shape == (n_samples, dim)
    for vector in result:
        assert np.allclose(np.linalg.norm(vector), 1, atol=1e-6)

    # Test case 2: n_samples = 3, dim = 3
    n_samples = 3
    dim = 3
    result = qs.sample_haar_vectors(n_samples, dim)
    assert result.shape == (n_samples, dim)
    for vector in result:
        assert np.allclose(np.linalg.norm(vector), 1, atol=1e-6)

    # Test case 3: n_samples = 4, dim = 4
    n_samples = 4
    dim = 4
    result = qs.sample_haar_vectors(n_samples, dim)
    assert result.shape == (n_samples, dim)
    for vector in result:
        assert np.allclose(np.linalg.norm(vector), 1, atol=1e-6)

    print("All test cases passed!")


test_sample_haar_vectors()


def test_sample_simplex_points():
    # Test case 1: n_samples = 2, dim = 2
    n_samples = 2
    dim = 2
    result = qs.sample_simplex_points(n_samples, dim)
    assert result.shape == (n_samples, dim)
    for point in result:
        assert np.allclose(np.sum(point), 1, atol=1e-6)

    # Test case 2: n_samples = 3, dim = 3
    n_samples = 3
    dim = 3
    result = qs.sample_simplex_points(n_samples, dim)
    assert result.shape == (n_samples, dim)
    for point in result:
        assert np.allclose(np.sum(point), 1, atol=1e-6)

    # Test case 3: n_samples = 4, dim = 4
    n_samples = 4
    dim = 4
    result = qs.sample_simplex_points(n_samples, dim)
    assert result.shape == (n_samples, dim)
    for point in result:
        assert np.allclose(np.sum(point), 1, atol=1e-6)

    print("All test cases passed!")


test_sample_simplex_points()


def test_combine():
    # Test case 1
    n_samples = 2
    dims = 2
    probabilities = qs.sample_simplex_points(n_samples, dims)
    unitaries = qs.sample_haar_unitaries(n_samples, dims)
    result = qs.combine(probabilities, unitaries)
    assert result.shape == (n_samples, dims, dims)
    for rho in result:
        assert np.allclose(rho, rho.conj().T, atol=1e-6)
        assert np.allclose(np.trace(rho), 1, atol=1e-6)

    # Test case 2
    n_samples = 3
    dims = 3
    probabilities = qs.sample_simplex_points(n_samples, dims)
    unitaries = qs.sample_haar_unitaries(n_samples, dims)
    result = qs.combine(probabilities, unitaries)
    assert result.shape == (n_samples, dims, dims)
    for rho in result:
        assert np.allclose(rho, rho.conj().T, atol=1e-6)
        assert np.allclose(np.trace(rho), 1, atol=1e-6)

    print("All test cases passed!")

    # Test case 2
    n_samples = 4
    dims = 4
    probabilities = qs.sample_simplex_points(n_samples, dims)
    unitaries = qs.sample_haar_unitaries(n_samples, dims)
    result = qs.combine(probabilities, unitaries)
    assert result.shape == (n_samples, dims, dims)
    for rho in result:
        assert np.allclose(rho, rho.conj().T, atol=1e-6)
        assert np.allclose(np.trace(rho), 1, atol=1e-6)

    print("All test cases passed!")


test_combine()


def test_sample_density_matrices():
    # Test case 1: n_samples = 2, dim = 2, method = 'natural'
    n_samples = 2
    dim = 2
    method = 'natural'
    result = qs.sample_density_matrices(n_samples, dim, method)
    assert result.shape == (n_samples, dim, dim)
    for rho in result:
        assert np.allclose(rho, rho.conj().T, atol=1e-6)
        assert np.allclose(np.trace(rho), 1, atol=1e-6)

    # Test case 2: n_samples = 3, dim = 3, method = 'ginibre'
    n_samples = 3
    dim = 3
    method = 'ginibre'
    result = qs.sample_density_matrices(n_samples, dim, method)
    assert result.shape == (n_samples, dim, dim)
    for rho in result:
        assert np.allclose(rho, rho.conj().T, atol=1e-6)
        assert np.allclose(np.trace(rho), 1, atol=1e-6)

    # Test case 3: n_samples = 4, dim = 4, method = 'natural'
    n_samples = 4
    dim = 4
    method = 'natural'
    result = qs.sample_density_matrices(n_samples, dim, method)
    assert result.shape == (n_samples, dim, dim)
    for rho in result:
        assert np.allclose(rho, rho.conj().T, atol=1e-6)
        assert np.allclose(np.trace(rho), 1, atol=1e-6)

    # Test case 4: n_samples = 5, dim = 5, method = 'ginibre'
    n_samples = 5
    dim = 5
    method = 'ginibre'
    result = qs.sample_density_matrices(n_samples, dim, method)
    assert result.shape == (n_samples, dim, dim)
    for rho in result:
        assert np.allclose(rho, rho.conj().T, atol=1e-6)
        assert np.allclose(np.trace(rho), 1, atol=1e-6)

    # Test case 5: Check for assertion error when an unknown method is provided
    n_samples = 2
    dim = 2
    method = 'unknown'
    try:
        result = qs.sample_density_matrices(n_samples, dim, method)
    except AssertionError as e:
        assert str(
            e) == f"Unknown method {method}. Choose from 'natural' or 'ginibre'."

    print("All test cases passed!")


test_sample_density_matrices()
