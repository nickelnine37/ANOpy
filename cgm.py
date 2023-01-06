import numpy as np
from typing import Callable


def conjugate_gradient(A: Callable[[np.ndarray], np.ndarray],
                       b: np.ndarray,
                       x0: np.ndarray = None,
                       Phi: Callable[[np.ndarray], np.ndarray] = None,
                       PhiT: Callable[[np.ndarray], np.ndarray] = None,
                       max_iter=20000,
                       reltol=1e-8,
                       verbose=False) -> np.ndarray:
    """

    Use the conjugate gradient method to solve the linear system Ax = b. Optionally, a symmetric
    preconditioner Phi can be specified, such that the new preconditioned problem is

    (Φ.T A Φ) (inv(Φ) x) = Φ.T b

    The arguments A, Phi and PhiT are functions which take a 1D numpy array and return an array of
    the same length. This means an optimised computation can be performed where possible.

    Parameters
    ----------
    A               The preconditioned coefficient matrix, as a function which performs the operation x -> (Φ.T A Φ) x
    b               The b parameter in the original problem
    x0              An optional initial guess for x
    Phi             The right symmetric preconditioner, as a funnction x -> Φ x
    PhiT            The left symmetric preconditioner, as a funnction x -> Φ.T x
    max_iter        Maximum number of iterations
    reltol          The relative tolerance level
    verbose         If True prints some extra information

    Returns
    -------
    x               The result of solving the linear system
    """

    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0

    if PhiT is not None:
        b = PhiT(b)

    r = b - A(x)

    d = r
    res_new = (r ** 2).sum()
    res0 = res_new

    its = 0

    while its < max_iter and res_new > (reltol ** 2 * res0):

        its += 1

        Ad = A(d)

        alpha = res_new / (d * Ad).sum()

        x += alpha * d

        # periodically rescale
        if its % 50 == 0:
            r = b - A(x)
            d = r

        else:
            r -= alpha * Ad

        res_old = res_new
        res_new = (r ** 2).sum()
        d = r + d * res_new / res_old

    if verbose:
        if its == max_iter:
            print(f'Warning: failed to converge in {its} iterations')
        else:
            print(f'Completed in {its} iterations')

    if Phi is None:
        return x

    else:
        return Phi(x)
