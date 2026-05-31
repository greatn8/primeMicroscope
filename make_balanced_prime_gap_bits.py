import math

LIMIT = 200_000_000
OUTFILE = "prime_gap_balanced_bits.bin"

def sieve(n):
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"

    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            start = i * i
            is_prime[start:n+1:i] = b"\x00" * (((n - start) // i) + 1)

    return is_prime

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

sorted_gaps = sorted(gaps)
median_gap = sorted_gaps[len(sorted_gaps) // 2]

above = sum(1 for g in gaps if g > median_gap)
equal = sum(1 for g in gaps if g == median_gap)

target_ones = len(gaps) // 2
need_equal_as_one = target_ones - above

print("Median gap:", median_gap)
print("Above median:", above)
print("Equal median:", equal)
print("Median gaps used as ones:", need_equal_as_one)

byte = 0
bit_pos = 0
written_bits = 0
ones = 0
equal_seen = 0

with open(OUTFILE, "wb") as out:
    for g in gaps:
        bit = 0

        if g > median_gap:
            bit = 1
        elif g == median_gap:
            equal_seen += 1

            before = ((equal_seen - 1) * need_equal_as_one) // equal
            after = (equal_seen * need_equal_as_one) // equal

            if after > before:
                bit = 1

        if bit:
            ones += 1

        byte |= bit << bit_pos
        bit_pos += 1
        written_bits += 1

        if bit_pos == 8:
            out.write(bytes([byte]))
            byte = 0
            bit_pos = 0

    if bit_pos > 0:
        out.write(bytes([byte]))

print("Created", OUTFILE)
print("Bits written:", written_bits)
print("Ones:", ones)
print("Zeros:", written_bits - ones)
print("Ones ratio:", ones / written_bits)
