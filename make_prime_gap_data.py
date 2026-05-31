import struct
import math

LIMIT = 50_000_000
OUTFILE = "prime_gaps.bin"

def sieve(n):
    is_prime = bytearray(b"\x01") * (n + 1)
    is_prime[0:2] = b"\x00\x00"

    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            start = i * i
            step = i
            is_prime[start:n+1:step] = b"\x00" * (((n - start) // step) + 1)

    return is_prime

print("Generating primes...")
is_prime = sieve(LIMIT)

print("Writing gaps...")
last_prime = None
count = 0

with open(OUTFILE, "wb") as out:
    for n in range(2, LIMIT + 1):
        if is_prime[n]:
            if last_prime is not None:
                gap = n - last_prime
                out.write(struct.pack("<I", gap))
                count += 1
            last_prime = n

print("Created", OUTFILE)
print("Prime gaps written:", count)
