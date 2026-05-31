import math

LIMIT = 200_000_000
OUTFILE = "prime_gap_updown_bits.bin"

def sieve(n):
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"

    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start:n+1:step] = b"\x00" * (((n - start) // step) + 1)

    return is_prime

print("Generating primes up to", LIMIT)
is_prime = sieve(LIMIT)

last_prime = None
last_gap = None

bits = []
byte = 0
bit_pos = 0
written_bits = 0

print("Writing up/down gap signal...")

with open(OUTFILE, "wb") as out:
    for n in range(2, LIMIT + 1):
        if is_prime[n]:
            if last_prime is not None:
                gap = n - last_prime

                if last_gap is not None:
                    if gap > last_gap:
                        bit = 1
                    else:
                        bit = 0

                    byte |= bit << bit_pos
                    bit_pos += 1
                    written_bits += 1

                    if bit_pos == 8:
                        out.write(bytes([byte]))
                        byte = 0
                        bit_pos = 0

                last_gap = gap

            last_prime = n

    if bit_pos > 0:
        out.write(bytes([byte]))

print("Created", OUTFILE)
print("Bits written:", written_bits)
print("Approx bytes:", written_bits // 8)
