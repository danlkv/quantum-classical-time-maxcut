import numpy as np

PHI = 0.5*(1+np.sqrt(5))

def fib_space(start, end, total, phi=PHI):
    len = end-start
    if total == 1:
        return np.array([start])
    rightcnt = round(total/(1+phi)) # smaller part to the right
    if rightcnt<1:
        right = []
        rightcnt = 1
    right = np.linspace(start+len/2, end, rightcnt-1, endpoint=False)
    left = fib_space(start, start+len/2, total-rightcnt, phi=phi)
    return np.concatenate((left, right, [end]))

def generate_x_log(start, end, total, phi=PHI):
    seq = fib_space(start, end, total-1, phi)
    return np.append(seq, end)


def test_generate_x_log():
    s = generate_x_log(0, 20, 21)
    print(s)
    assert len(s) == 21

    s = generate_x_log(0, 20, 21, phi=4)
    print(s)
    assert len(s) == 21

