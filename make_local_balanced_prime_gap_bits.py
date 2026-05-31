import math

LIMIT = 200_000_000
BLOCK_GAPS = 8192
OUTFILE = "prime_gap_local_balanced_bits.bin"

def sieve(n):
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"

    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            start = i * i
            is_prime[start:n+1:i] = b"\x00" * (((n - start) // i) + 1)

    return is_prime

def write_bits(out, bits, state):
    byte, bit_pos, written, ones = state

    for bit in bits:
        if bit:
            ones += 1
            byte |= 1 << bit_pos

        bit_pos += 1
        written += 1

        if bit_pos == 8:
            out.write(bytes([byte]))
            byte = 0
            bit_pos = 0

    return byte, bit_pos, written, ones

print("Generating primes up to", LIMIT)
is_prime = sieve(LIMIT)

gaps = []
last_prime = None

for n in range(2, LIMIT + 1):
    if is_prime[n]:
        if last_prime is not None:
            gaps.append(n - last_prime)
        last_prime = n

print("Prime gaps:", len(gaps))
print("Block size:", BLOCK_GAPS)

state = (0, 0, 0, 0)

with open(OUTFILE, "wb") as out:
    for start in range(0, len(gaps), BLOCK_GAPS):
        block = gaps[start:start + BLOCK_GAPS]

        if len(block) < 16:
            continue

        ranked = sorted(range(len(block)), key=lambda i: (block[i], i))
        bits = [0] * len(block)

        for idx in ranked[len(block) // 2:]:
            bits[idx] = 1

        state = write_bits(out, bits, state)

    byte, bit_pos, written, ones = state

    if bit_pos > 0:
        out.write(bytes([byte]))

print("Created", OUTFILE)
print("Bits written:", written)
print("Ones:", ones)
print("Zeros:", written - ones)
print("Ones ratio:", ones / written)
