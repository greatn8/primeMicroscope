/*
 * prime2.cu
 *
 * CUDA iterative prime pattern explorer.
 *
 * This version does NOT just scan larger ranges.
 * It marks primes once, then tests many different pattern lenses:
 *
 *   - pair correlations: p and p+d for d = 2,4,6,...,210
 *   - known prime tuple-style offset groups
 *   - cluster-style offset groups
 *
 * Outputs:
 *   - iter_000_novelty.bmp
 *   - iter_000_winner.bmp
 *   - iter_000_pattern_*.bmp
 *   - iter_000_top_tiles.csv
 *   - iter_000_top_patterns.csv
 *
 * Compile:
 *   nvcc -O3 -std=c++17 -arch=sm_80 prime2.cu -o prime2
 *
 * Run:
 *   ./prime2 1000000000 50000000 2 128 128
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

const double PI_VALUE = 3.141592653589793238462643383279502884;


#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)             \
                      << " at line " << __LINE__ << std::endl;                 \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct RGB {
    unsigned char r, g, b;
};

struct PatternBank {
    std::vector<std::string> names;
    std::vector<int> starts;
    std::vector<int> lengths;
    std::vector<int> max_offsets;
    std::vector<int> flat_offsets;
};

struct TileRank {
    int tile = 0;
    double novelty = 0.0;
    int winner = 0;
    double winner_z = 0.0;
    uint64_t range_start = 0;
    uint64_t range_end = 0;
};

struct PatternRank {
    int pattern = 0;
    double max_z = 0.0;
    unsigned int top_count = 0;
    int top_tile = 0;
    double mean = 0.0;
    double stddev = 1.0;
};

__device__ bool is_prime_gpu(uint64_t n, const int *small_primes, int prime_count)
{
    if (n < 2) return false;
    if (n == 2) return true;
    if ((n & 1ULL) == 0ULL) return false;

    for (int i = 0; i < prime_count; i++) {
        uint64_t p = (uint64_t) small_primes[i];
        if (p * p > n) break;
        if (n % p == 0ULL) return n == p;
    }

    return true;
}

__global__ void mark_primes_kernel(
    uint64_t base_start,
    uint64_t odd_count,
    const int *small_primes,
    int prime_count,
    unsigned char *prime_flags
) {
    uint64_t tid = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t) gridDim.x * blockDim.x;

    for (uint64_t idx = tid; idx < odd_count; idx += stride) {
        uint64_t n = base_start + 2ULL * idx;
        prime_flags[idx] = is_prime_gpu(n, small_primes, prime_count) ? 1 : 0;
    }
}

__global__ void count_patterns_kernel(
    const unsigned char *prime_flags,
    uint64_t odd_count,
    int tile_count,
    uint64_t bin_size,
    const int *pattern_starts,
    const int *pattern_lengths,
    const int *pattern_max_offsets,
    const int *flat_offsets,
    int pattern_count,
    unsigned int *counts
) {
    uint64_t job = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_jobs = (uint64_t) pattern_count * tile_count;
    uint64_t stride = (uint64_t) gridDim.x * blockDim.x;

    for (; job < total_jobs; job += stride) {
        int pattern_id = (int) (job / tile_count);
        int tile_id = (int) (job % tile_count);

        uint64_t begin = (uint64_t) tile_id * bin_size;
        uint64_t end = begin + bin_size;
        if (end > odd_count) end = odd_count;

        int start = pattern_starts[pattern_id];
        int len = pattern_lengths[pattern_id];
        int max_off = pattern_max_offsets[pattern_id];

        unsigned int local_count = 0;

        if (begin < end) {
            for (uint64_t i = begin; i < end; i++) {
                if (i + (uint64_t) max_off >= odd_count) break;

                bool ok = true;

                for (int k = 0; k < len; k++) {
                    int off = flat_offsets[start + k];

                    if (!prime_flags[i + (uint64_t) off]) {
                        ok = false;
                        break;
                    }
                }

                if (ok) {
                    local_count++;
                }
            }
        }

        counts[(uint64_t) pattern_id * tile_count + tile_id] = local_count;
    }
}

std::vector<int> make_small_primes(uint64_t limit)
{
    std::vector<unsigned char> sieve(limit + 1, 1);
    std::vector<int> primes;

    sieve[0] = 0;
    if (limit >= 1) sieve[1] = 0;

    for (uint64_t i = 2; i * i <= limit; i++) {
        if (sieve[i]) {
            for (uint64_t j = i * i; j <= limit; j += i) {
                sieve[j] = 0;
            }
        }
    }

    for (uint64_t i = 2; i <= limit; i++) {
        if (sieve[i]) primes.push_back((int) i);
    }

    return primes;
}

void add_pattern(PatternBank &bank, const std::string &name,
                 const std::vector<int> &even_offsets)
{
    bank.names.push_back(name);
    bank.starts.push_back((int) bank.flat_offsets.size());
    bank.lengths.push_back((int) even_offsets.size());

    int max_unit = 0;

    for (int off : even_offsets) {
        int unit = off / 2;
        bank.flat_offsets.push_back(unit);
        if (unit > max_unit) max_unit = unit;
    }

    bank.max_offsets.push_back(max_unit);
}

const int MAX_PATTERN_LENSES = 100000;

bool is_admissible_pattern(const std::vector<int> &offsets)
{
    int small_mods[] = {3, 5, 7, 11, 13};

    for (int q : small_mods) {
        bool seen[13] = {false};

        for (int off : offsets) {
            int r = off % q;
            if (r < 0) r += q;
            seen[r] = true;
        }

        bool covers_all_residues = true;

        for (int r = 0; r < q; r++) {
            if (!seen[r]) {
                covers_all_residues = false;
                break;
            }
        }

        if (covers_all_residues) {
            return false;
        }
    }

    return true;
}

void add_pattern_if_admissible(PatternBank &bank,
                               const std::string &name,
                               const std::vector<int> &offsets)
{
    if ((int) bank.names.size() >= MAX_PATTERN_LENSES) {
        return;
    }

    if (is_admissible_pattern(offsets)) {
        add_pattern(bank, name, offsets);
    }
}

PatternBank build_pattern_bank()
{
    PatternBank bank;

    // ------------------------------------------------------------
    // 1. Important known dense prime-constellation-style patterns
    // ------------------------------------------------------------
    add_pattern_if_admissible(bank, "triplet_0_2_6", {0, 2, 6});
    add_pattern_if_admissible(bank, "triplet_0_4_6", {0, 4, 6});

    add_pattern_if_admissible(bank, "quad_0_2_6_8", {0, 2, 6, 8});
    add_pattern_if_admissible(bank, "quad_0_4_6_10", {0, 4, 6, 10});

    add_pattern_if_admissible(bank, "quint_0_2_6_8_12", {0, 2, 6, 8, 12});
    add_pattern_if_admissible(bank, "quint_0_4_6_10_12", {0, 4, 6, 10, 12});

    add_pattern_if_admissible(bank, "sextuplet_0_4_6_10_12_16",
                              {0, 4, 6, 10, 12, 16});

    add_pattern_if_admissible(bank, "cluster_0_2_6_8_12_18",
                              {0, 2, 6, 8, 12, 18});

    add_pattern_if_admissible(bank, "cluster_0_4_6_10_12_18",
                              {0, 4, 6, 10, 12, 18});

    add_pattern_if_admissible(bank, "cluster_0_6_12_18_24",
                              {0, 6, 12, 18, 24});

    // ------------------------------------------------------------
    // 2. Many pair-correlation patterns
    // ------------------------------------------------------------
    for (int d = 2; d <= 5000; d += 2) {
        std::ostringstream name;
        name << "pair_offset_" << d;
        add_pattern_if_admissible(bank, name.str(), {0, d});
    }

    // ------------------------------------------------------------
    // 3. Wider chain/arithmetic patterns
    // ------------------------------------------------------------
    for (int step = 30; step <= 3000; step += 30) {
        {
            std::ostringstream name;
            name << "chain_0_" << step << "_" << (2 * step);
            add_pattern_if_admissible(bank, name.str(),
                                      {0, step, 2 * step});
        }

        {
            std::ostringstream name;
            name << "chain_0_" << step << "_" << (2 * step)
                 << "_" << (3 * step);
            add_pattern_if_admissible(bank, name.str(),
                                      {0, step, 2 * step, 3 * step});
        }
    }

    // ------------------------------------------------------------
    // 4. Automatically generated triplet lenses
    // ------------------------------------------------------------
    for (int a = 2; a <= 180; a += 2) {
        for (int b = a + 2; b <= 600; b += 2) {
            std::ostringstream name;
            name << "triplet_0_" << a << "_" << b;
            add_pattern_if_admissible(bank, name.str(), {0, a, b});

            if ((int) bank.names.size() >= MAX_PATTERN_LENSES) {
                return bank;
            }
        }
    }

    // ------------------------------------------------------------
    // 5. Automatically generated quad lenses
    // ------------------------------------------------------------
    for (int a = 2; a <= 80; a += 2) {
        for (int b = a + 2; b <= 180; b += 2) {
            for (int c = b + 2; c <= 360; c += 2) {
                std::ostringstream name;
                name << "quad_0_" << a << "_" << b << "_" << c;
                add_pattern_if_admissible(bank, name.str(), {0, a, b, c});

                if ((int) bank.names.size() >= MAX_PATTERN_LENSES) {
                    return bank;
                }
            }
        }
    }

    return bank;
}

bool write_bmp_24(const std::string &filename, int width, int height,
                  const std::vector<RGB> &pixels)
{
    int row_stride = width * 3;
    int padding = (4 - (row_stride % 4)) % 4;
    int padded_row_size = row_stride + padding;
    int pixel_data_size = padded_row_size * height;
    int file_size = 54 + pixel_data_size;

    unsigned char file_header[14] = {
        'B', 'M',
        0, 0, 0, 0,
        0, 0,
        0, 0,
        54, 0, 0, 0
    };

    unsigned char info_header[40] = {
        40, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        1, 0,
        24, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0x13, 0x0B, 0, 0,
        0x13, 0x0B, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0
    };

    file_header[2] = (unsigned char) file_size;
    file_header[3] = (unsigned char) (file_size >> 8);
    file_header[4] = (unsigned char) (file_size >> 16);
    file_header[5] = (unsigned char) (file_size >> 24);

    info_header[4] = (unsigned char) width;
    info_header[5] = (unsigned char) (width >> 8);
    info_header[6] = (unsigned char) (width >> 16);
    info_header[7] = (unsigned char) (width >> 24);

    info_header[8] = (unsigned char) height;
    info_header[9] = (unsigned char) (height >> 8);
    info_header[10] = (unsigned char) (height >> 16);
    info_header[11] = (unsigned char) (height >> 24);

    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    out.write((char *) file_header, 14);
    out.write((char *) info_header, 40);

    std::vector<unsigned char> row(padded_row_size);

    for (int y = height - 1; y >= 0; y--) {
        int k = 0;

        for (int x = 0; x < width; x++) {
            RGB p = pixels[y * width + x];
            row[k++] = p.b;
            row[k++] = p.g;
            row[k++] = p.r;
        }

        while (k < padded_row_size) row[k++] = 0;
        out.write((char *) row.data(), padded_row_size);
    }

    return true;
}

double clamp01(double x)
{
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

RGB heat_color(double t)
{
    t = clamp01(t);

    double r, g, b;

    if (t < 0.25) {
        double u = t / 0.25;
        r = 0;
        g = u * 255.0;
        b = 128.0 + u * 127.0;
    } else if (t < 0.5) {
        double u = (t - 0.25) / 0.25;
        r = 0;
        g = 255.0;
        b = 255.0 - u * 255.0;
    } else if (t < 0.75) {
        double u = (t - 0.5) / 0.25;
        r = u * 255.0;
        g = 255.0;
        b = 0;
    } else {
        double u = (t - 0.75) / 0.25;
        r = 255.0;
        g = 255.0 - u * 255.0;
        b = 0;
    }

    return {(unsigned char) r, (unsigned char) g, (unsigned char) b};
}

RGB pattern_color(int id)
{
    unsigned int x = (unsigned int) id * 2654435761u;

    unsigned char r = (unsigned char) (80 + (x & 127));
    unsigned char g = (unsigned char) (80 + ((x >> 8) & 127));
    unsigned char b = (unsigned char) (80 + ((x >> 16) & 127));

    return {r, g, b};
}

int pattern_family_id(const std::string &name)
{
    if (name.find("pair_") == 0) return 0;
    if (name.find("triplet_") == 0) return 1;
    if (name.find("quad_") == 0) return 2;
    if (name.find("quint_") == 0) return 3;
    if (name.find("sextuplet_") == 0) return 4;
    if (name.find("cluster_") == 0) return 5;
    if (name.find("chain_") == 0) return 6;

    return 7;
}

RGB family_color(int family)
{
    switch (family) {
        case 0: return {80, 160, 255};   // pair: blue
        case 1: return {80, 255, 160};   // triplet: green
        case 2: return {255, 220, 80};   // quad: yellow
        case 3: return {255, 140, 80};   // quint: orange
        case 4: return {255, 80, 120};   // sextuplet: pink/red
        case 5: return {180, 80, 255};   // cluster: purple
        case 6: return {80, 255, 255};   // chain: cyan
        default: return {180, 180, 180}; // other: grey
    }
}

std::vector<RGB> winner_family_image(const PatternBank &bank,
                                     const std::vector<int> &winner,
                                     const std::vector<double> &winner_z,
                                     int width,
                                     int height)
{
    std::vector<RGB> img(width * height);

    double z_cap = 12.0;

    for (int i = 0; i < width * height; i++) {
        int pattern_id = winner[i];
        int family = pattern_family_id(bank.names[pattern_id]);

        RGB base = family_color(family);

        double strength = std::log1p(winner_z[i]) / std::log1p(z_cap);
        strength = clamp01(strength);

        double brightness = 0.15 + 0.85 * strength;

        img[i].r = (unsigned char) std::round(base.r * brightness);
        img[i].g = (unsigned char) std::round(base.g * brightness);
        img[i].b = (unsigned char) std::round(base.b * brightness);
    }

    return img;
}

std::vector<RGB> spiral_metric_image(const std::vector<double> &metric,
                                     int tile_count,
                                     int size)
{
    std::vector<RGB> img(size * size, RGB{0, 0, 0});

    if (tile_count <= 0) {
        return img;
    }

    double maxv = 0.0;

    for (int i = 0; i < tile_count; i++) {
        if (metric[i] > maxv) {
            maxv = metric[i];
        }
    }

    if (maxv <= 0.0) {
        maxv = 1.0;
    }

    double cx = (size - 1) / 2.0;
    double cy = (size - 1) / 2.0;
    double maxr = size * 0.47;

    double turns = 32.0;

    // Increase this if you want larger dots.
    int dot_radius = 2;

    // Anything below this relative strength becomes very dim.
    double threshold = 0.08;

    for (int t = 0; t < tile_count; t++) {
        double u = ((double) t + 0.5) / (double) tile_count;

        double r = maxr * std::sqrt(u);
        double theta = 2.0 * PI_VALUE * turns * u;

        int x0 = (int) std::round(cx + r * std::cos(theta));
        int y0 = (int) std::round(cy + r * std::sin(theta));

        double raw = metric[t] / maxv;
        raw = clamp01(raw);

        // Log scaling makes mid-strength novelty easier to see.
        double strength = std::log1p(20.0 * raw) / std::log1p(20.0);
        strength = clamp01(strength);

        if (strength < threshold) {
            strength *= 0.20;
        }

        RGB colour = heat_color(strength);

        for (int dy = -dot_radius; dy <= dot_radius; dy++) {
            for (int dx = -dot_radius; dx <= dot_radius; dx++) {
                int x = x0 + dx;
                int y = y0 + dy;

                if (x < 0 || x >= size || y < 0 || y >= size) {
                    continue;
                }

                double dist = std::sqrt((double) (dx * dx + dy * dy));

                if (dist > dot_radius) {
                    continue;
                }

                double fade = 1.0 - (dist / (double) (dot_radius + 1));
                fade = clamp01(fade);

                RGB &old = img[y * size + x];

                old.r = (unsigned char) std::min(
                    255,
                    (int) old.r + (int) std::round(colour.r * fade)
                );

                old.g = (unsigned char) std::min(
                    255,
                    (int) old.g + (int) std::round(colour.g * fade)
                );

                old.b = (unsigned char) std::min(
                    255,
                    (int) old.b + (int) std::round(colour.b * fade)
                );
            }
        }
    }

    return img;
}


std::vector<RGB> novelty_wheel_image(const std::vector<double> &novelty,
                                     uint64_t base_start,
                                     uint64_t bin_size,
                                     int valid_tile_count,
                                     int width,
                                     int height,
                                     uint64_t modulus)
{
    std::vector<RGB> img(width * height, RGB{0, 0, 0});

    if (valid_tile_count <= 0) {
        return img;
    }

    double maxv = 0.0;

    for (int i = 0; i < valid_tile_count; i++) {
        if (novelty[i] > maxv) {
            maxv = novelty[i];
        }
    }

    if (maxv <= 0.0) {
        maxv = 1.0;
    }

    int dot_radius = 1;

    for (int t = 0; t < valid_tile_count; t++) {
        uint64_t tile_start = base_start + 2ULL * (uint64_t)t * bin_size;

        double progress = (double)t / (double)(valid_tile_count - 1);
        uint64_t residue = tile_start % modulus;

        int x0 = (int)std::round(progress * (width - 1));
        int y0 = (int)std::round(((double)residue / (double)(modulus - 1)) * (height - 1));

        double raw = novelty[t] / maxv;
        raw = clamp01(raw);

        double strength = std::log1p(20.0 * raw) / std::log1p(20.0);
        strength = clamp01(strength);

        RGB colour = heat_color(strength);

        for (int dy = -dot_radius; dy <= dot_radius; dy++) {
            for (int dx = -dot_radius; dx <= dot_radius; dx++) {
                int x = x0 + dx;
                int y = y0 + dy;

                if (x < 0 || x >= width || y < 0 || y >= height) {
                    continue;
                }

                RGB &old = img[y * width + x];

                old.r = (unsigned char)std::max((int)old.r, (int)colour.r);
                old.g = (unsigned char)std::max((int)old.g, (int)colour.g);
                old.b = (unsigned char)std::max((int)old.b, (int)colour.b);
            }
        }
    }

    return img;
}


std::vector<RGB> novelty_torus_image(const std::vector<double> &novelty,
                                     uint64_t base_start,
                                     uint64_t bin_size,
                                     int valid_tile_count,
                                     int size,
                                     uint64_t mod_x,
                                     uint64_t mod_y)
{
    std::vector<RGB> img(size * size, RGB{0, 0, 0});

    if (valid_tile_count <= 0) {
        return img;
    }

    double maxv = 0.0;

    for (int i = 0; i < valid_tile_count; i++) {
        if (novelty[i] > maxv) {
            maxv = novelty[i];
        }
    }

    if (maxv <= 0.0) {
        maxv = 1.0;
    }

    int dot_radius = 2;

    for (int t = 0; t < valid_tile_count; t++) {
        uint64_t tile_start = base_start + 2ULL * (uint64_t)t * bin_size;

        uint64_t rx = tile_start % mod_x;
        uint64_t ry = tile_start % mod_y;

        int x0 = (int)std::round(((double)rx / (double)(mod_x - 1)) * (size - 1));
        int y0 = (int)std::round(((double)ry / (double)(mod_y - 1)) * (size - 1));

        double raw = novelty[t] / maxv;
        raw = clamp01(raw);

        double strength = std::log1p(20.0 * raw) / std::log1p(20.0);
        strength = clamp01(strength);

        RGB colour = heat_color(strength);

        for (int dy = -dot_radius; dy <= dot_radius; dy++) {
            for (int dx = -dot_radius; dx <= dot_radius; dx++) {
                int x = x0 + dx;
                int y = y0 + dy;

                if (x < 0 || x >= size || y < 0 || y >= size) {
                    continue;
                }

                double dist = std::sqrt((double)(dx * dx + dy * dy));

                if (dist > dot_radius) {
                    continue;
                }

                RGB &old = img[y * size + x];

                old.r = (unsigned char)std::max((int)old.r, (int)colour.r);
                old.g = (unsigned char)std::max((int)old.g, (int)colour.g);
                old.b = (unsigned char)std::max((int)old.b, (int)colour.b);
            }
        }
    }

    return img;
}

std::vector<RGB> spiral_winner_family_image(const PatternBank &bank,
                                            const std::vector<int> &winner,
                                            const std::vector<double> &winner_z,
                                            int tile_count,
                                            int size)
{
    std::vector<RGB> img(size * size, RGB{0, 0, 0});

    if (tile_count <= 0) {
        return img;
    }

    double cx = (size - 1) / 2.0;
    double cy = (size - 1) / 2.0;
    double maxr = size * 0.47;

    double turns = 32.0;
    double z_cap = 16.0;

    int dot_radius = 2;

    for (int t = 0; t < tile_count; t++) {
        double u = ((double) t + 0.5) / (double) tile_count;

        double r = maxr * std::sqrt(u);
        double theta = 2.0 * PI_VALUE * turns * u;

        int x0 = (int) std::round(cx + r * std::cos(theta));
        int y0 = (int) std::round(cy + r * std::sin(theta));

        int pattern_id = winner[t];
        int family = pattern_family_id(bank.names[pattern_id]);

        RGB base = family_color(family);

        double strength = std::log1p(winner_z[t]) / std::log1p(z_cap);
        strength = clamp01(strength);

        double brightness = 0.10 + 0.90 * strength;

        RGB colour = {
            (unsigned char) std::round(base.r * brightness),
            (unsigned char) std::round(base.g * brightness),
            (unsigned char) std::round(base.b * brightness)
        };

        for (int dy = -dot_radius; dy <= dot_radius; dy++) {
            for (int dx = -dot_radius; dx <= dot_radius; dx++) {
                int x = x0 + dx;
                int y = y0 + dy;

                if (x < 0 || x >= size || y < 0 || y >= size) {
                    continue;
                }

                double dist = std::sqrt((double) (dx * dx + dy * dy));

                if (dist > dot_radius) {
                    continue;
                }

                double fade = 1.0 - (dist / (double) (dot_radius + 1));
                fade = clamp01(fade);

                RGB &old = img[y * size + x];

                old.r = (unsigned char) std::min(
                    255,
                    (int) old.r + (int) std::round(colour.r * fade)
                );

                old.g = (unsigned char) std::min(
                    255,
                    (int) old.g + (int) std::round(colour.g * fade)
                );

                old.b = (unsigned char) std::min(
                    255,
                    (int) old.b + (int) std::round(colour.b * fade)
                );
            }
        }
    }

    return img;
}

std::string clean_name(const std::string &s)
{
    std::string out;

    for (char c : s) {
        if ((c >= 'a' && c <= 'z') ||
            (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9')) {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }

    return out;
}

std::string iter_prefix(int iter)
{
    std::ostringstream oss;
    oss << "iter_" << std::setw(3) << std::setfill('0') << iter << "_";
    return oss.str();
}
void write_pattern_occurrences_csv(
    const std::string &filename,
    const std::vector<unsigned char> &prime_flags,
    uint64_t base_start,
    uint64_t odd_count,
    uint64_t bin_size,
    int tile_id,
    int pattern_id,
    const PatternBank &bank,
    int max_rows
);

std::vector<RGB> metric_image(const std::vector<double> &values,
                              int width, int height)
{
    std::vector<RGB> img(width * height);

    double mn = *std::min_element(values.begin(), values.end());
    double mx = *std::max_element(values.begin(), values.end());

    if (mx - mn < 1e-12) {
        for (auto &p : img) p = {0, 0, 0};
        return img;
    }

    for (size_t i = 0; i < values.size(); i++) {
        double t = (values[i] - mn) / (mx - mn);
        img[i] = heat_color(t);
    }

    return img;
}

std::vector<RGB> winner_image(const std::vector<int> &winner,
                              const std::vector<double> &winner_z,
                              int width, int height)
{
    std::vector<RGB> img(width * height);

    // Cap the brightness scale so one extreme tile does not make
    // everything else nearly black.
    double z_cap = 12.0;

    for (int i = 0; i < width * height; i++) {
        RGB base = pattern_color(winner[i]);

        // Log scaling makes medium-strong winners visible.
        double strength = std::log1p(winner_z[i]) / std::log1p(z_cap);
        strength = clamp01(strength);

        // Keep weak tiles visible but still darker than strong tiles.
        double brightness = 0.15 + 0.85 * strength;

        img[i].r = (unsigned char) std::round(base.r * brightness);
        img[i].g = (unsigned char) std::round(base.g * brightness);
        img[i].b = (unsigned char) std::round(base.b * brightness);
    }

    return img;
}

void mean_std_for_pattern(const std::vector<unsigned int> &counts,
                          int pattern_id,
                          int tile_count,
                          int valid_tile_count,
                          double &mean,
                          double &stddev)
{
    uint64_t offset = (uint64_t) pattern_id * tile_count;

    if (valid_tile_count <= 0) {
        mean = 0.0;
        stddev = 1.0;
        return;
    }

    double sum = 0.0;

    for (int t = 0; t < valid_tile_count; t++) {
        sum += counts[offset + t];
    }

    mean = sum / valid_tile_count;

    double ss = 0.0;

    for (int t = 0; t < valid_tile_count; t++) {
        double d = counts[offset + t] - mean;
        ss += d * d;
    }

    stddev = std::sqrt(ss / valid_tile_count);

    if (stddev < 1e-9) {
        stddev = 1.0;
    }
}

void analyse_and_output(
    int iter,
    uint64_t base_start,
    uint64_t odd_count,
    int width,
    int height,
    uint64_t bin_size,
    const PatternBank &bank,
    const std::vector<unsigned int> &counts,
    const std::vector<unsigned char> &prime_flags
) {
    int tile_count = width * height;
    int pattern_count = (int) bank.names.size();

        int valid_tile_count = (int) std::min<uint64_t>(
        (uint64_t) tile_count,
        (odd_count + bin_size - 1ULL) / bin_size
    );

    if (valid_tile_count <= 0) {
        std::cerr << "No valid tiles to analyse.\n";
        return;
    }
    std::vector<double> novelty(tile_count, 0.0);
    std::vector<int> winner(tile_count, 0);
    std::vector<double> winner_z(tile_count, 0.0);

    std::vector<PatternRank> pattern_ranks;

    for (int p = 0; p < pattern_count; p++) {
        double mean, stddev;
        mean_std_for_pattern(counts, p, tile_count, valid_tile_count, mean, stddev);

        PatternRank rank;
        rank.pattern = p;
        rank.mean = mean;
        rank.stddev = stddev;

        uint64_t offset = (uint64_t) p * tile_count;

        for (int t = 0; t < valid_tile_count; t++) {
            double z = std::fabs(((double) counts[offset + t] - mean) / stddev);

            novelty[t] += z;

            if (z > winner_z[t]) {
                winner_z[t] = z;
                winner[t] = p;
            }

            if (z > rank.max_z) {
                rank.max_z = z;
                rank.top_count = counts[offset + t];
                rank.top_tile = t;
            }
        }

        pattern_ranks.push_back(rank);
    }

    for (double &x : novelty) {
        x /= pattern_count;
    }

    std::vector<TileRank> tile_ranks;

    for (int t = 0; t < valid_tile_count; t++) {
        uint64_t begin = (uint64_t) t * bin_size;
        uint64_t end = std::min(begin + bin_size, odd_count);

        TileRank tr;
        tr.tile = t;
        tr.novelty = novelty[t];
        tr.winner = winner[t];
        tr.winner_z = winner_z[t];
        tr.range_start = base_start + 2ULL * begin;
        tr.range_end = (end > begin)
            ? base_start + 2ULL * (end - 1ULL)
            : tr.range_start;

        tile_ranks.push_back(tr);
    }

    std::sort(tile_ranks.begin(), tile_ranks.end(),
              [](const TileRank &a, const TileRank &b) {
                  return a.novelty > b.novelty;
              });

    std::sort(pattern_ranks.begin(), pattern_ranks.end(),
              [](const PatternRank &a, const PatternRank &b) {
                  return a.max_z > b.max_z;
              });
    std::sort(pattern_ranks.begin(), pattern_ranks.end(),
            [](const PatternRank &a, const PatternRank &b) {
                return a.max_z > b.max_z;
            });

    std::string prefix = iter_prefix(iter);

    int top_occurrence_files = 80;
    int max_occurrence_rows = 10000;

    // Write exact occurrences for the top pattern lenses.
    // These files answer: "Where exactly did this top pattern occur?"
    for (int i = 0; i < top_occurrence_files && i < (int) pattern_ranks.size(); i++) {
        const PatternRank &pr = pattern_ranks[i];

        std::ostringstream fname;
        fname << prefix
            << "occurrence_pattern_"
            << std::setw(2) << std::setfill('0') << i
            << "_"
            << clean_name(bank.names[pr.pattern])
            << ".csv";

        write_pattern_occurrences_csv(
            fname.str(),
            prime_flags,
            base_start,
            odd_count,
            bin_size,
            pr.top_tile,
            pr.pattern,
            bank,
            max_occurrence_rows
        );
    }

    // Write exact occurrences for the top unusual tiles using each tile's winner pattern.
    // These files answer: "What exact p-values explain this unusual tile?"
    for (int i = 0; i < top_occurrence_files && i < (int) tile_ranks.size(); i++) {
        const TileRank &tr = tile_ranks[i];

        std::ostringstream fname;
        fname << prefix
            << "occurrence_tile_"
            << std::setw(2) << std::setfill('0') << i
            << "_"
            << clean_name(bank.names[tr.winner])
            << ".csv";

        write_pattern_occurrences_csv(
            fname.str(),
            prime_flags,
            base_start,
            odd_count,
            bin_size,
            tr.tile,
            tr.winner,
            bank,
            max_occurrence_rows
        );
    }
    
    int family_counts[8] = {0};

    for (int t = 0; t < valid_tile_count; t++) {
        int pattern_id = winner[t];
        int family = pattern_family_id(bank.names[pattern_id]);

        if (family >= 0 && family < 8) {
            family_counts[family]++;
        }
    }

    std::string family_names[8] = {
        "pair",
        "triplet",
        "quad",
        "quint",
        "sextuplet",
        "cluster",
        "chain",
        "other"
    };

    std::ofstream family_csv(prefix + "winner_family_summary.csv");
    family_csv << "family,count,percentage\n";

    for (int f = 0; f < 8; f++) {
        double percentage = 100.0 * (double) family_counts[f] / (double) valid_tile_count;

        family_csv << family_names[f] << ","
                << family_counts[f] << ","
                << percentage << "\n";
    }

    std::cout << "Winner family summary:\n";
    for (int f = 0; f < 8; f++) {
        double percentage = 100.0 * (double) family_counts[f] / (double) valid_tile_count;

        std::cout << "  " << family_names[f]
                << ": " << family_counts[f]
                << " tiles (" << percentage << "%)\n";
    }          

    write_bmp_24(prefix + "novelty.bmp", width, height,
                 metric_image(novelty, width, height));

    write_bmp_24(prefix + "winner.bmp", width, height,
                 winner_image(winner, winner_z, width, height));

    write_bmp_24(prefix + "winner_family.bmp", width, height,
                winner_family_image(bank, winner, winner_z, width, height));    
                
    int spiral_size = 1536;

    write_bmp_24(prefix + "novelty_spiral.bmp",
                spiral_size, spiral_size,
                spiral_metric_image(novelty, valid_tile_count, spiral_size));

    write_bmp_24(prefix + "winner_family_spiral.bmp",
                spiral_size, spiral_size,
                spiral_winner_family_image(bank, winner, winner_z,
                                            valid_tile_count, spiral_size));

    int math_view_size = 1024;

    write_bmp_24(prefix + "novelty_wheel_2310.bmp",
                math_view_size,
                math_view_size,
                novelty_wheel_image(novelty,
                                    base_start,
                                    bin_size,
                                    valid_tile_count,
                                    math_view_size,
                                    math_view_size,
                                    2310ULL));

    write_bmp_24(prefix + "novelty_wheel_30030.bmp",
                math_view_size,
                math_view_size,
                novelty_wheel_image(novelty,
                                    base_start,
                                    bin_size,
                                    valid_tile_count,
                                    math_view_size,
                                    math_view_size,
                                    30030ULL));

    write_bmp_24(prefix + "novelty_torus_210_2310.bmp",
                math_view_size,
                math_view_size,
                novelty_torus_image(novelty,
                                    base_start,
                                    bin_size,
                                    valid_tile_count,
                                    math_view_size,
                                    210ULL,
                                    2310ULL));
                                                                                
    std::ofstream tiles_csv(prefix + "top_tiles.csv");
    tiles_csv << "rank,range_start,range_end,novelty,winner_pattern,winner_z\n";

    for (int i = 0; i < 50 && i < (int) tile_ranks.size(); i++) {
        const TileRank &t = tile_ranks[i];

        tiles_csv << (i + 1) << ","
                  << t.range_start << ","
                  << t.range_end << ","
                  << t.novelty << ","
                  << bank.names[t.winner] << ","
                  << t.winner_z << "\n";
    }

    std::ofstream patterns_csv(prefix + "top_patterns.csv");
    patterns_csv << "rank,pattern,max_z,top_count,top_tile,mean,stddev,top_range_start,top_range_end\n";

    int images_to_write = std::min(24, (int) pattern_ranks.size());

    for (int i = 0; i < (int) pattern_ranks.size(); i++) {
        const PatternRank &pr = pattern_ranks[i];

        int t = pr.top_tile;
        uint64_t begin = (uint64_t) t * bin_size;
        uint64_t end = std::min(begin + bin_size, odd_count);

        uint64_t rs = base_start + 2ULL * begin;
        uint64_t re = (end > begin)
            ? base_start + 2ULL * (end - 1ULL)
            : rs;

        patterns_csv << (i + 1) << ","
                    << bank.names[pr.pattern] << ","
                    << pr.max_z << ","
                    << pr.top_count << ","
                    << pr.top_tile << ","
                    << pr.mean << ","
                    << pr.stddev << ","
                    << rs << ","
                    << re << "\n";

        if (i < images_to_write) {
            std::vector<double> metric(tile_count);
            uint64_t off = (uint64_t) pr.pattern * tile_count;

            for (int j = 0; j < tile_count; j++) {
                metric[j] = counts[off + j];
            }

            std::ostringstream fname;
            fname << prefix
                  << "pattern_"
                  << std::setw(2) << std::setfill('0') << i
                  << "_"
                  << clean_name(bank.names[pr.pattern])
                  << ".bmp";

            write_bmp_24(fname.str(), width, height,
                         metric_image(metric, width, height));
        }
    }

    std::cout << "Top 10 unusual pattern lenses:\n";

    for (int i = 0; i < 10 && i < (int) pattern_ranks.size(); i++) {
        const PatternRank &pr = pattern_ranks[i];

        std::cout << "  #" << (i + 1)
                << " " << bank.names[pr.pattern]
                << " max_z=" << pr.max_z
                << " top_count=" << pr.top_count
                << " mean=" << pr.mean
                << " std=" << pr.stddev
                << "\n";
    }

    std::cout << "Top 5 unusual tiles:\n";

    for (int i = 0; i < 5 && i < (int) tile_ranks.size(); i++) {
        const TileRank &tr = tile_ranks[i];

        std::cout << "  #" << (i + 1)
                  << " [" << tr.range_start << ", " << tr.range_end << "]"
                  << " novelty=" << tr.novelty
                  << " winner=" << bank.names[tr.winner]
                  << " winner_z=" << tr.winner_z
                  << "\n";
    }
    std::cout << "Saved " << prefix << "novelty.bmp, "
            << prefix << "winner.bmp, "
            << prefix << "winner_family.bmp, "
            << prefix << "top_tiles.csv, "
            << prefix << "top_patterns.csv, top pattern BMPs, "
            << "and exact occurrence CSVs.\n";
}

void write_pattern_occurrences_csv(
    const std::string &filename,
    const std::vector<unsigned char> &prime_flags,
    uint64_t base_start,
    uint64_t odd_count,
    uint64_t bin_size,
    int tile_id,
    int pattern_id,
    const PatternBank &bank,
    int max_rows
) {
    if (tile_id < 0 || pattern_id < 0) {
        return;
    }

    uint64_t begin = (uint64_t) tile_id * bin_size;
    uint64_t end = begin + bin_size;

    if (begin >= odd_count) {
        return;
    }

    if (end > odd_count) {
        end = odd_count;
    }

    int pattern_start = bank.starts[pattern_id];
    int pattern_len = bank.lengths[pattern_id];
    int max_off = bank.max_offsets[pattern_id];

    std::ofstream out(filename);

    out << "pattern," << bank.names[pattern_id] << "\n";
    out << "tile_id," << tile_id << "\n";
    out << "range_start," << (base_start + 2ULL * begin) << "\n";
    out << "range_end," << (base_start + 2ULL * (end - 1ULL)) << "\n";

    out << "base_p";

    for (int k = 0; k < pattern_len; k++) {
        int off_units = bank.flat_offsets[pattern_start + k];
        int actual_offset = off_units * 2;
        out << ",p_plus_" << actual_offset;
    }

    out << "\n";

    int rows_written = 0;

    for (uint64_t i = begin; i < end; i++) {
        if (i + (uint64_t) max_off >= odd_count) {
            break;
        }

        bool ok = true;

        for (int k = 0; k < pattern_len; k++) {
            int off_units = bank.flat_offsets[pattern_start + k];

            if (!prime_flags[i + (uint64_t) off_units]) {
                ok = false;
                break;
            }
        }

        if (ok) {
            uint64_t p = base_start + 2ULL * i;

            out << p;

            for (int k = 0; k < pattern_len; k++) {
                int off_units = bank.flat_offsets[pattern_start + k];
                uint64_t actual_offset = 2ULL * (uint64_t) off_units;
                out << "," << (p + actual_offset);
            }

            out << "\n";

            rows_written++;

            if (max_rows > 0 && rows_written >= max_rows) {
                break;
            }
        }
    }
}

struct GapMotifStats {
    uint64_t total = 0;
    std::unordered_map<int, int> tile_counts;
};

struct GapMotifRank {
    std::string motif;
    uint64_t total = 0;
    int best_tile = 0;
    int top_count = 0;
    double mean = 0.0;
    double stddev = 0.0;
    double z = 0.0;
    uint64_t range_start = 0;
    uint64_t range_end = 0;
};

std::vector<int> parse_gap_motif_key(const std::string &key)
{
    std::vector<int> gaps;

    if (key.find("gap_") != 0) {
        return gaps;
    }

    std::stringstream ss(key.substr(4));
    std::string part;

    while (std::getline(ss, part, '_')) {
        gaps.push_back(std::stoi(part));
    }

    return gaps;
}

void write_gap_motif_occurrences(
    const std::string &filename,
    const std::vector<uint64_t> &primes,
    const std::string &motif,
    int best_tile,
    uint64_t base_start,
    uint64_t bin_size
) {
    std::vector<int> gaps = parse_gap_motif_key(motif);

    if (gaps.empty()) {
        return;
    }

    std::ofstream out(filename);

    out << "motif," << motif << "\n";
    out << "best_tile," << best_tile << "\n";
    out << "base_p";

    for (int i = 0; i <= (int)gaps.size(); i++) {
        out << ",prime_" << i;
    }

    out << "\n";

    for (size_t i = 0; i + gaps.size() < primes.size(); i++) {
        uint64_t p = primes[i];

        int tile = (int)((p - base_start) / (2ULL * bin_size));

        if (tile != best_tile) {
            continue;
        }

        bool ok = true;

        for (size_t g = 0; g < gaps.size(); g++) {
            uint64_t actual_gap = primes[i + g + 1] - primes[i + g];

            if (actual_gap != (uint64_t)gaps[g]) {
                ok = false;
                break;
            }
        }

        if (ok) {
            out << p;

            for (size_t g = 0; g <= gaps.size(); g++) {
                out << "," << primes[i + g];
            }

            out << "\n";
        }
    }
}

void analyse_gap_motifs_csv(
    int iter,
    uint64_t base_start,
    uint64_t odd_count,
    uint64_t bin_size,
    int valid_tile_count,
    const std::vector<unsigned char> &prime_flags,
    int motif_len,
    int max_total_span,
    int top_n
) {
    std::vector<uint64_t> primes;
    primes.reserve(odd_count / 20);

    for (uint64_t i = 0; i < odd_count; i++) {
        if (prime_flags[i]) {
            primes.push_back(base_start + 2ULL * i);
        }
    }

    std::unordered_map<std::string, GapMotifStats> stats;

    for (size_t i = 0; i + motif_len < primes.size(); i++) {
        std::vector<uint64_t> gaps;
        uint64_t total_span = 0;

        bool ok = true;

        for (int g = 0; g < motif_len; g++) {
            uint64_t gap = primes[i + g + 1] - primes[i + g];
            total_span += gap;

            if (total_span > (uint64_t)max_total_span) {
                ok = false;
                break;
            }

            gaps.push_back(gap);
        }

        if (!ok) {
            continue;
        }

        uint64_t p = primes[i];
        int tile = (int)((p - base_start) / (2ULL * bin_size));

        if (tile < 0 || tile >= valid_tile_count) {
            continue;
        }

        std::ostringstream key;
        key << "gap";

        for (uint64_t gap : gaps) {
            key << "_" << gap;
        }

        std::string motif = key.str();

        GapMotifStats &s = stats[motif];
        s.total++;
        s.tile_counts[tile]++;
    }

    std::vector<GapMotifRank> ranks;

    for (const auto &entry : stats) {
        const std::string &motif = entry.first;
        const GapMotifStats &s = entry.second;

        GapMotifRank r;
        r.motif = motif;
        r.total = s.total;

        double mean = (double)s.total / (double)valid_tile_count;
        double sumsq = 0.0;

        for (const auto &tc : s.tile_counts) {
            double c = (double)tc.second;
            sumsq += c * c;

            if (tc.second > r.top_count) {
                r.top_count = tc.second;
                r.best_tile = tc.first;
            }
        }

        double variance = (sumsq / (double)valid_tile_count) - mean * mean;

        if (variance < 0.0) {
            variance = 0.0;
        }

        double stddev = std::sqrt(variance);
        double z = 0.0;

        if (stddev > 0.0) {
            z = ((double)r.top_count - mean) / stddev;
        }

        r.mean = mean;
        r.stddev = stddev;
        r.z = z;
        r.range_start = base_start + 2ULL * (uint64_t)r.best_tile * bin_size;
        r.range_end = r.range_start + 2ULL * bin_size - 2ULL;

        ranks.push_back(r);
    }

    std::sort(ranks.begin(), ranks.end(),
              [](const GapMotifRank &a, const GapMotifRank &b) {
                  if (a.top_count != b.top_count) {
                      return a.top_count > b.top_count;
                  }

                  if (a.z != b.z) {
                      return a.z > b.z;
                  }

                  return a.total > b.total;
              });

    std::ostringstream prefix;
    prefix << "iter_" << std::setw(3) << std::setfill('0') << iter << "_";

    std::ostringstream filename;
    filename << prefix.str()
             << "gap_motifs_len"
             << motif_len
             << ".csv";

    std::ofstream out(filename.str());

    out << "rank,motif,total_count,top_count,best_tile,z,mean,stddev,range_start,range_end\n";

    int limit = std::min(top_n, (int)ranks.size());

    for (int i = 0; i < limit; i++) {
        const GapMotifRank &r = ranks[i];

        out << (i + 1) << ","
            << r.motif << ","
            << r.total << ","
            << r.top_count << ","
            << r.best_tile << ","
            << r.z << ","
            << r.mean << ","
            << r.stddev << ","
            << r.range_start << ","
            << r.range_end << "\n";

        if (i < 20) {
            std::ostringstream occ_name;
            occ_name << prefix.str()
                     << "gap_motif_occurrence_"
                     << std::setw(2) << std::setfill('0') << i
                     << "_"
                     << r.motif
                     << ".csv";

            write_gap_motif_occurrences(
                occ_name.str(),
                primes,
                r.motif,
                r.best_tile,
                base_start,
                bin_size
            );
        }
    }

    std::cout << "Saved " << filename.str()
              << " and top gap motif occurrence CSVs.\n";
}

int main(int argc, char **argv)
{
    uint64_t start = 1000000000ULL;
    uint64_t range_per_iter = 50000000ULL;
    int iterations = 2;
    int width = 128;
    int height = 128;

    if (argc >= 2) start = std::strtoull(argv[1], nullptr, 10);
    if (argc >= 3) range_per_iter = std::strtoull(argv[2], nullptr, 10);
    if (argc >= 4) iterations = std::atoi(argv[3]);
    if (argc >= 5) width = std::atoi(argv[4]);
    if (argc >= 6) height = std::atoi(argv[5]);

    if (iterations <= 0 || width <= 0 || height <= 0 || range_per_iter < 1000) {
        std::cerr << "Usage: ./prime2 START RANGE_PER_ITER ITERATIONS WIDTH HEIGHT\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    int device_id = 0;

    CHECK_CUDA(cudaGetDevice(&device_id));
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));

    PatternBank bank = build_pattern_bank();

    int pattern_count = (int) bank.names.size();

    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Pattern lenses: " << pattern_count << "\n";
    std::cout << "Range per iteration: " << range_per_iter << "\n";
    std::cout << "Iterations: " << iterations << "\n";
    std::cout << "Image size: " << width << "x" << height << "\n\n";

    int *dev_primes = nullptr;
    unsigned char *dev_flags = nullptr;
    int *dev_starts = nullptr;
    int *dev_lengths = nullptr;
    int *dev_max_offsets = nullptr;
    int *dev_flat_offsets = nullptr;
    unsigned int *dev_counts = nullptr;

    int tile_count = width * height;
    uint64_t total_count_slots = (uint64_t) pattern_count * tile_count;

    CHECK_CUDA(cudaMalloc((void **) &dev_starts,
                          bank.starts.size() * sizeof(int)));

    CHECK_CUDA(cudaMalloc((void **) &dev_lengths,
                          bank.lengths.size() * sizeof(int)));

    CHECK_CUDA(cudaMalloc((void **) &dev_max_offsets,
                          bank.max_offsets.size() * sizeof(int)));

    CHECK_CUDA(cudaMalloc((void **) &dev_flat_offsets,
                          bank.flat_offsets.size() * sizeof(int)));

    CHECK_CUDA(cudaMalloc((void **) &dev_counts,
                          total_count_slots * sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(dev_starts,
                          bank.starts.data(),
                          bank.starts.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dev_lengths,
                          bank.lengths.data(),
                          bank.lengths.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dev_max_offsets,
                          bank.max_offsets.data(),
                          bank.max_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dev_flat_offsets,
                          bank.flat_offsets.data(),
                          bank.flat_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    for (int iter = 0; iter < iterations; iter++) {
        uint64_t iter_start = start + (uint64_t) iter * range_per_iter;
        uint64_t iter_end = iter_start + range_per_iter;

        uint64_t base_start = (iter_start % 2ULL == 0ULL)
            ? iter_start + 1ULL
            : iter_start;

        uint64_t odd_count = ((iter_end - 1ULL) >= base_start)
            ? (((iter_end - 1ULL) - base_start) / 2ULL + 1ULL)
            : 0ULL;

        if (odd_count == 0) continue;

        uint64_t bin_size = (odd_count + tile_count - 1ULL) / tile_count;

        uint64_t max_value = iter_end + 1000ULL;
        uint64_t small_prime_limit =
            (uint64_t) std::sqrt((long double) max_value) + 2ULL;

        std::cout << "=== Iteration " << iter << " ===\n";
        std::cout << "Range: [" << iter_start << ", " << iter_end << ")\n";
        std::cout << "Odd candidates: " << odd_count << "\n";
        std::cout << "Tile bin size: " << bin_size << " odd candidates per pixel/tile\n";
        std::cout << "Generating small primes to " << small_prime_limit << "...\n";

        std::vector<int> host_primes = make_small_primes(small_prime_limit);

        CHECK_CUDA(cudaMalloc((void **) &dev_primes,
                              host_primes.size() * sizeof(int)));

        CHECK_CUDA(cudaMalloc((void **) &dev_flags,
                              odd_count * sizeof(unsigned char)));

        CHECK_CUDA(cudaMemcpy(dev_primes,
                              host_primes.data(),
                              host_primes.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

        int block_size = 256;
        int grid_size = prop.multiProcessorCount * 32;

        cudaEvent_t e1, e2;
        CHECK_CUDA(cudaEventCreate(&e1));
        CHECK_CUDA(cudaEventCreate(&e2));

        CHECK_CUDA(cudaEventRecord(e1));

        mark_primes_kernel<<<grid_size, block_size>>>(
            base_start,
            odd_count,
            dev_primes,
            (int) host_primes.size(),
            dev_flags
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(e2));
        CHECK_CUDA(cudaEventSynchronize(e2));

        float ms_prime = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_prime, e1, e2));

        std::cout << "Prime marking time: " << (ms_prime / 1000.0) << " sec\n";

        CHECK_CUDA(cudaMemset(dev_counts, 0,
                              total_count_slots * sizeof(unsigned int)));

        CHECK_CUDA(cudaEventRecord(e1));

        uint64_t pattern_jobs = (uint64_t) pattern_count * tile_count;
        int pattern_grid = (int) std::min<uint64_t>(
            (pattern_jobs + block_size - 1ULL) / block_size,
            (uint64_t) prop.multiProcessorCount * 64ULL
        );

        count_patterns_kernel<<<pattern_grid, block_size>>>(
            dev_flags,
            odd_count,
            tile_count,
            bin_size,
            dev_starts,
            dev_lengths,
            dev_max_offsets,
            dev_flat_offsets,
            pattern_count,
            dev_counts
        );

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaEventRecord(e2));
        CHECK_CUDA(cudaEventSynchronize(e2));

        float ms_patterns = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms_patterns, e1, e2));

        std::cout << "Pattern analysis time: " << (ms_patterns / 1000.0) << " sec\n";

        std::vector<unsigned int> host_counts(total_count_slots);

        CHECK_CUDA(cudaMemcpy(host_counts.data(),
                            dev_counts,
                            total_count_slots * sizeof(unsigned int),
                            cudaMemcpyDeviceToHost));

        std::vector<unsigned char> host_flags(odd_count);

        CHECK_CUDA(cudaMemcpy(host_flags.data(),
                            dev_flags,
                            odd_count * sizeof(unsigned char),
                            cudaMemcpyDeviceToHost));

        analyse_and_output(
            iter,
            base_start,
            odd_count,
            width,
            height,
            bin_size,
            bank,
            host_counts,
            host_flags
        );

        // New: analyse consecutive prime-gap motifs.
        // This looks for repeated local "words" in the actual prime gap sequence,
        // e.g. gap_6_4_98 means consecutive primes spaced +6, +4, +98.
        int valid_tile_count_for_gaps =
            (int)std::min<uint64_t>(
                (uint64_t)width * (uint64_t)height,
                (odd_count + bin_size - 1ULL) / bin_size
            );

        analyse_gap_motifs_csv(
            iter,
            base_start,
            odd_count,
            bin_size,
            valid_tile_count_for_gaps,
            host_flags,
            3,      // motif length: 3 gaps = 4 consecutive primes
            360,    // max total span considered
            300     // top rows to save
        );

        CHECK_CUDA(cudaFree(dev_primes));
        CHECK_CUDA(cudaFree(dev_flags));
        CHECK_CUDA(cudaEventDestroy(e1));
        CHECK_CUDA(cudaEventDestroy(e2));

        std::cout << "\n";
    }

    CHECK_CUDA(cudaFree(dev_starts));
    CHECK_CUDA(cudaFree(dev_lengths));
    CHECK_CUDA(cudaFree(dev_max_offsets));
    CHECK_CUDA(cudaFree(dev_flat_offsets));
    CHECK_CUDA(cudaFree(dev_counts));

    std::cout << "Done.\n";

    return EXIT_SUCCESS;
}