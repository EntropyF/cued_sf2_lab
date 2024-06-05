from typing import Tuple, NamedTuple, Optional
import numpy as np


def diagscan(N: int) -> np.ndarray:
    '''
    Generate diagonal scanning pattern

    Returns:
        A diagonal scanning index for a flattened NxN matrix

        The first entry in the matrix is assumed to be the DC coefficient
        and is therefore not included in the scan
    '''
    if N <= 1:
        raise ValueError('Cannot generate a scan pattern for a {}x{} matrix'.format(N, N))

    # Copied from matlab without accounting for indexing.
    slast = N + 1
    scan = [slast]
    while slast != N * N:
        while slast > N and slast % N != 0:
            slast = slast - (N - 1)
            scan.append(slast)
        if slast < N:
            slast = slast + 1
        else:
            slast = slast + N
        scan.append(slast)
        if slast == N * N:
            break
        while slast < (N * N - N + 1) and slast % N != 1:
            slast = slast + (N - 1)
            scan.append(slast)
        if slast == N * N:
            break
        if slast < (N * N - N + 1):
            slast = slast + N
        else:
            slast = slast + 1
        scan.append(slast)
    # Python indexing
    return np.array(scan) - 1

def row_wise_scan(N: int) -> np.ndarray:
    '''
    Generate row-wise scanning pattern

    Returns:
        A row-wise scanning index for a flattened NxN matrix
    '''
    if N <= 1:
        raise ValueError('Cannot generate a scan pattern for a {}x{} matrix'.format(N, N))
    
    scan = []
    for row in range(N):
        for col in range(N):
            scan.append(row * N + col)
    
    return np.array(scan)

def column_wise_scan(N: int) -> np.ndarray:
    '''
    Generate column-wise scanning pattern

    Returns:
        A column-wise scanning index for a flattened NxN matrix
    '''
    if N <= 1:
        raise ValueError('Cannot generate a scan pattern for a {}x{} matrix'.format(N, N))
    
    scan = []
    for col in range(N):
        for row in range(N):
            scan.append(row * N + col)
    
    return np.array(scan)

def zigzag_scan(N: int) -> np.ndarray:
    '''
    Generate zigzag scanning pattern

    Returns:
        A zigzag scanning index for a flattened NxN matrix
    '''
    if N <= 1:
        raise ValueError('Cannot generate a scan pattern for a {}x{} matrix'.format(N, N))
    
    scan = np.zeros(N * N, dtype=int)
    index = 0
    
    for sum in range(2 * N - 1):
        if sum % 2 == 0:
            for i in range(sum + 1):
                if i < N and sum - i < N:
                    scan[index] = i * N + (sum - i)
                    index += 1
        else:
            for i in range(sum + 1):
                if i < N and sum - i < N:
                    scan[index] = (sum - i) * N + i
                    index += 1
    
    return scan
