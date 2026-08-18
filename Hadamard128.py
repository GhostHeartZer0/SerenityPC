import numpy as np

def generatesylvesterhadamard(k):
    """Generates a Sylvester Hadamard matrix of order 2^k."""
    size = 2**k
    matrix = np.ones((size, size))
    for i in range(size):
        for j in range(size):
            # The entry is -1 if the number of set bits in (i AND j) is odd
            if bin(i & j).count('1') % 2 != 0:
                matrix[i, j] = -1
            else:
                matrix[i, j] = 1
    return matrix

# Generate H_128
H128 = generatesylvesterhadamard(7)

# Verification: H  H.T must equal 128  I
identitycheck = np.dot(H128, H_128.T)
ishaldamard = np.allclose(identitycheck, 128 * np.eye(128))

print(f"Matrix Shape: {H_128.shape}")
print(f"Is valid Hadamard matrix? {is_haldamard}")