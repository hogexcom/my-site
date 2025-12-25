import math

def inv_tran(tran):
    return (-tran[0], -tran[1])

def eq_tran(tran, edge1, edge2, A, b):
    for idx, i in enumerate(edge1):
        A[i * 2][i * 2] += 1
        A[i * 2][edge2[idx] * 2] += -1
        b[i * 2] -= tran[0]
        A[i * 2 + 1][i * 2 + 1] += 1
        A[i * 2 + 1][edge2[idx] * 2 + 1] += -1
        b[i * 2 + 1] -= tran[1]
    pass

def eq_rot(theta, edge1, edge2, origin, A, b):
    for idx, i in enumerate(edge1):
        A[i * 2][i * 2] += math.cos(theta)
        A[i * 2][i * 2 + 1] += -1 * math.sin(theta)
        A[i * 2][origin * 2] += 1 - math.cos(theta)
        A[i * 2][origin * 2 + 1] += math.sin(theta)
        A[i * 2][edge2[idx] * 2] += -1
        A[i * 2 + 1][i * 2] += math.sin(theta)
        A[i * 2 + 1][i * 2 + 1] += math.cos(theta)
        A[i * 2 + 1][origin * 2] += -1 * math.sin(theta)
        A[i * 2 + 1][origin * 2 + 1] += 1 - math.cos(theta)
        A[i * 2 + 1][edge2[idx] * 2 + 1] += -1
        pass