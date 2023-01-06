import numpy as np
from numpy.linalg import eigh

filter_functions = {'inverse': lambda lamL, beta: (1 + beta * lamL) ** -1,
                    'exponential': lambda lamL, beta: np.exp(-beta * lamL),
                    'ReLu': lambda lamL, beta: np.maximum(1 - beta * lamL, 0),
                    'sigmoid': lambda lamL, beta: 2 * np.exp(-beta * lamL) * (1 + np.exp(-beta * lamL)) ** -1,
                    'cosine': lambda lamL, beta: np.cos(lamL * np.pi / (2 * lamL.max())) ** np.exp(beta),
                    # 'cut-off': lambda lamL, beta: (lamL <= 1 / beta).astype(float) if beta != 0 else np.ones_like(lamL)}
                    'cut-off': lambda lamL, beta: 1 - 1 / (1 + np.exp(-20 * (lamL - beta)))}


def vec(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        return X
    elif X.ndim == 2:
        return X.T.reshape(-1)
    else:
        raise ValueError



def mat(x: np.ndarray, shape: tuple = None, like: np.ndarray = None) -> np.ndarray:
    if shape is None and like is None:
        raise ValueError('Pass either shape or like')

    if shape is not None and like is not None:
        raise ValueError('Pass only one of shape or like')

    if shape is not None:
        if len(shape) != 2:
            raise ValueError(f'shape parameter must be length 2, but it is {shape}')

    else:
        shape = like.shape
        if len(shape) != 2:
            raise ValueError(f'shape of the passed array must be length 2, but it is {shape}')

    if x.ndim == 2:
        if any(s == 1 for s in x.shape):
            return x.reshape(-1).reshape((shape[1], shape[0])).T
        else:
            return x

    elif x.ndim == 1:
        return x.reshape((shape[1], shape[0])).T

    else:
        raise ValueError(f'Cannot vectorise x with {x.ndim} dimensions')
        
        
def get_random_graph(N, p=0.5):
    A = np.triu(np.random.choice([0, 1], size=(N, N), replace=True, p=[1 - p, p]), 1)
    A = A + A.T
    L = np.diag(A.sum(0)) - A
    return A, L


def get_chain_graph(N):
    A = np.zeros((N, N))
    A[range(0, N-1), range(1, N)] = 1
    A = A + A.T
    L = np.diag(A.sum(0)) - A
    return A, L



def generate_toy_data(N, T, gamma, beta=None, function='exponential', random_graph=False, seed=0, p=0.5):

    np.random.seed(seed)

    Y = np.random.normal(size=(N, T))
    S = np.random.choice([0, 1], p=[1 - p, p], replace=True, size=(N, T))
    S_ = 1 - S
    Y = Y * S

    if random_graph:
        AT, LT = get_random_graph(T)
        AN, LN = get_random_graph(N)

    else:
        AT, LT = get_chain_graph(T)
        AN, LN = get_chain_graph(N)

    lamLT, UT = eigh(LT)
    lamLN, UN = eigh(LN)

    if beta is None:
        #         beta = 5 / max([max(lamLN), max(lamLT)])
        beta = {'inverse': 10, 'exponential': 1.5, 'ReLu': 0.8, 'sigmoid': 1.5, 'cosine': 3.5, 'cut-off': 1.5}[function]

    eta = filter_functions[function]
    Lam = lamLN[:, None] + lamLT[None, :]
    G = eta(Lam, beta).astype(float)

    J = G ** 2 / (G ** 2 + gamma)

    A_ = (AT.sum(0)[None, :] + AN.sum(0)[:, None])

    return T, N, gamma, beta, eta, Y, S, S_, UT, UN, AT, AN, LT, LN, A_, J, G, Lam

