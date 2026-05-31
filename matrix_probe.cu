#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <string>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <unordered_map>

#define BLOCK_SIZE 256

__device__ __forceinline__ int bit_at(const unsigned char* data, uint64_t bit_index) {
    uint64_t byte_index = bit_index >> 3;
    int bit_offset = bit_index & 7;
    return (data[byte_index] >> bit_offset) & 1;
}

__global__ void count_ones_kernel(const unsigned char* data, uint64_t nbytes, uint64_t* block_counts) {
    __shared__ uint64_t shared[BLOCK_SIZE];

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
    uint64_t local = 0;

    for (uint64_t i = tid; i < nbytes; i += stride) {
        local += __popc((unsigned int)data[i]);
    }

    shared[threadIdx.x] = local;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = shared[0];
    }
}

__global__ void autocorr_kernel(
    const unsigned char* data,
    uint64_t nbits,
    int max_lag,
    uint64_t* partial_counts
) {
    __shared__ uint64_t shared[BLOCK_SIZE];

    int lag = blockIdx.y + 1;
    int block_id = blockIdx.x;

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = blockDim.x * gridDim.x;
    uint64_t limit = nbits - lag;

    uint64_t local_equal = 0;

    for (uint64_t i = tid; i < limit; i += stride) {
        int a = bit_at(data, i);
        int b = bit_at(data, i + lag);

        if (a == b) {
            local_equal++;
        }
    }

    shared[threadIdx.x] = local_equal;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_counts[(uint64_t)(lag - 1) * gridDim.x + block_id] = shared[0];
    }
}

struct Hit {
    int lag;
    double z;
    double ratio;
    int repeats;
    int family_support;
};

uint64_t rng64(uint64_t& x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

std::string now_string() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);

    std::stringstream ss;
    ss << std::put_time(std::localtime(&t), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void check_cuda(cudaError_t result, const std::string& label) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << label << ": "
                  << cudaGetErrorString(result) << "\n";
        exit(1);
    }
}

int family_support(const std::vector<Hit>& hits, int lag) {
    int support = 0;

    for (const auto& h : hits) {
        if (h.lag == lag) {
            continue;
        }

        int small = std::min(lag, h.lag);
        int large = std::max(lag, h.lag);

        if (small > 0 && large % small == 0 && large / small <= 16) {
            support++;
        }
    }

    return support;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage:\n";
        std::cout << "  ./matrix_probe data.bin [max_lag] [window_mb] [save_z] [amazing_z] [min_lag] [max_abs_ones_z]\n\n";
        std::cout << "Example:\n";
        std::cout << "  ./matrix_probe prime_signal.bin 1024 1 8.0 12.0 9 10.0\n";
        return 1;
    }

    std::string filename = argv[1];

    int max_lag = 1024;
    int window_mb = 1;
    double save_z = 8.0;
    double amazing_z = 12.0;
    int min_lag = 9;
    double max_abs_ones_z = 10.0;

    if (argc >= 3) max_lag = std::stoi(argv[2]);
    if (argc >= 4) window_mb = std::stoi(argv[3]);
    if (argc >= 5) save_z = std::stod(argv[4]);
    if (argc >= 6) amazing_z = std::stod(argv[5]);
    if (argc >= 7) min_lag = std::stoi(argv[6]);
    if (argc >= 8) max_abs_ones_z = std::stod(argv[7]);

    if (window_mb < 1) {
        std::cerr << "window_mb must be at least 1.\n";
        return 1;
    }

    const int blocks = 512;
    uint64_t window_bytes = (uint64_t)window_mb * 1024ULL * 1024ULL;

    std::ifstream test_file(filename, std::ios::binary | std::ios::ate);
    if (!test_file) {
        std::cerr << "Could not open input file.\n";
        return 1;
    }

    uint64_t file_size = (uint64_t)test_file.tellg();
    test_file.close();

    if (file_size < window_bytes) {
        std::cerr << "File is smaller than the chosen window size.\n";
        std::cerr << "File size: " << file_size << " bytes\n";
        std::cerr << "Window size: " << window_bytes << " bytes\n";
        return 1;
    }

    std::ofstream hits_file("interesting_hits.csv", std::ios::app);
    std::ofstream best_file("best_seen.txt", std::ios::app);
    std::ofstream checkpoint_file("checkpoint.txt", std::ios::app);

    hits_file << "time,epoch,offset,window_bytes,lag,z,equal_ratio,ones_z,repeat_count,family_support,reason\n";
    hits_file.flush();

    unsigned char* d_data = nullptr;
    uint64_t* d_ones_blocks = nullptr;
    uint64_t* d_partial = nullptr;

    check_cuda(cudaMalloc(&d_data, window_bytes), "cudaMalloc d_data");
    check_cuda(cudaMalloc(&d_ones_blocks, blocks * sizeof(uint64_t)), "cudaMalloc d_ones_blocks");
    check_cuda(cudaMalloc(&d_partial, (uint64_t)blocks * max_lag * sizeof(uint64_t)), "cudaMalloc d_partial");

    std::vector<unsigned char> host_data(window_bytes);
    std::vector<uint64_t> host_ones(blocks);
    std::vector<uint64_t> host_partial((uint64_t)blocks * max_lag);

    std::unordered_map<int, int> lag_repeat_count;

    uint64_t seed = 0x9e3779b97f4a7c15ULL;
    uint64_t epoch = 0;
    uint64_t skipped_biased = 0;

    double best_abs_z = 0.0;
    int best_lag = 0;
    uint64_t best_offset = 0;

    std::cout << "matrix_probe started\n";
    std::cout << "file: " << filename << "\n";
    std::cout << "file size: " << file_size << " bytes\n";
    std::cout << "window: " << window_mb << " MB\n";
    std::cout << "max lag: " << max_lag << "\n";
    std::cout << "min lag: " << min_lag << "\n";
    std::cout << "save z: " << save_z << "\n";
    std::cout << "amazing z: " << amazing_z << "\n";
    std::cout << "max abs ones z: " << max_abs_ones_z << "\n";
    std::cout << "running until stopped...\n\n";

    while (true) {
        epoch++;

        uint64_t range = file_size - window_bytes;
        uint64_t offset = 0;

        if (range > 0) {
            offset = rng64(seed) % range;
            offset -= offset % 8;
        }

        std::ifstream file(filename, std::ios::binary);
        file.seekg(offset, std::ios::beg);
        file.read(reinterpret_cast<char*>(host_data.data()), window_bytes);

        if (!file) {
            std::cerr << "Read failed at offset " << offset << "\n";
            continue;
        }

        check_cuda(cudaMemcpy(d_data, host_data.data(), window_bytes, cudaMemcpyHostToDevice), "copy data");

        count_ones_kernel<<<blocks, BLOCK_SIZE>>>(d_data, window_bytes, d_ones_blocks);
        check_cuda(cudaDeviceSynchronize(), "count ones sync");

        check_cuda(cudaMemcpy(host_ones.data(), d_ones_blocks, blocks * sizeof(uint64_t), cudaMemcpyDeviceToHost), "copy ones");

        uint64_t ones = 0;
        for (uint64_t x : host_ones) {
            ones += x;
        }

        uint64_t nbits = window_bytes * 8ULL;
        double ones_expected = nbits / 2.0;
        double ones_sd = std::sqrt(nbits * 0.25);
        double ones_z = (ones - ones_expected) / ones_sd;

        if (std::fabs(ones_z) > max_abs_ones_z) {
            skipped_biased++;

            checkpoint_file << now_string()
                            << " epoch=" << epoch
                            << " offset=" << offset
                            << " skipped_biased=1"
                            << " ones_z=" << ones_z
                            << " total_skipped_biased=" << skipped_biased
                            << "\n";
            checkpoint_file.flush();

            std::cout << "epoch " << epoch
                      << " | offset " << offset
                      << " | skipped biased window"
                      << " | ones_z " << ones_z
                      << " | skipped total " << skipped_biased
                      << "\n";

            continue;
        }

        dim3 grid(blocks, max_lag);
        autocorr_kernel<<<grid, BLOCK_SIZE>>>(d_data, nbits, max_lag, d_partial);
        check_cuda(cudaDeviceSynchronize(), "autocorr sync");

        check_cuda(
            cudaMemcpy(
                host_partial.data(),
                d_partial,
                (uint64_t)blocks * max_lag * sizeof(uint64_t),
                cudaMemcpyDeviceToHost
            ),
            "copy autocorr"
        );

        std::vector<Hit> hits;

        for (int lag = min_lag; lag <= max_lag; lag++) {
            uint64_t equal_count = 0;

            for (int b = 0; b < blocks; b++) {
                equal_count += host_partial[(uint64_t)(lag - 1) * blocks + b];
            }

            uint64_t comparisons = nbits - lag;
            double expected = comparisons / 2.0;
            double sd = std::sqrt(comparisons * 0.25);

            double z = (equal_count - expected) / sd;
            double ratio = (double)equal_count / (double)comparisons;

            if (std::fabs(z) >= save_z) {
                lag_repeat_count[lag]++;
                hits.push_back({lag, z, ratio, lag_repeat_count[lag], 0});
            }

            if (std::fabs(z) > best_abs_z) {
                best_abs_z = std::fabs(z);
                best_lag = lag;
                best_offset = offset;

                best_file << now_string()
                          << " new_best epoch=" << epoch
                          << " offset=" << offset
                          << " lag=" << lag
                          << " z=" << z
                          << " abs_z=" << best_abs_z
                          << " ratio=" << ratio
                          << " ones_z=" << ones_z
                          << "\n";
                best_file.flush();
            }
        }

        for (auto& h : hits) {
            h.family_support = family_support(hits, h.lag);
        }

        std::sort(hits.begin(), hits.end(), [](const Hit& a, const Hit& b) {
            if (a.family_support != b.family_support) {
                return a.family_support > b.family_support;
            }

            return std::fabs(a.z) > std::fabs(b.z);
        });

        for (const auto& h : hits) {
            std::string reason = "strong_high_lag";

            if (std::fabs(h.z) >= amazing_z && h.repeats >= 5 && h.family_support >= 2) {
                reason = "AMAZING_REPEAT_FAMILY";
            } else if (h.repeats >= 5 && h.family_support >= 2) {
                reason = "repeat_family_candidate";
            } else if (h.repeats >= 5) {
                reason = "repeated_single_lag";
            }

            hits_file << now_string() << ","
                      << epoch << ","
                      << offset << ","
                      << window_bytes << ","
                      << h.lag << ","
                      << h.z << ","
                      << h.ratio << ","
                      << ones_z << ","
                      << h.repeats << ","
                      << h.family_support << ","
                      << reason
                      << "\n";

            if (reason == "AMAZING_REPEAT_FAMILY") {
                std::ofstream amazing("amazing_candidate.txt", std::ios::app);
                amazing << now_string() << "\n";
                amazing << "epoch=" << epoch << "\n";
                amazing << "offset=" << offset << "\n";
                amazing << "window_bytes=" << window_bytes << "\n";
                amazing << "lag=" << h.lag << "\n";
                amazing << "z=" << h.z << "\n";
                amazing << "equal_ratio=" << h.ratio << "\n";
                amazing << "ones_z=" << ones_z << "\n";
                amazing << "repeat_count=" << h.repeats << "\n";
                amazing << "family_support=" << h.family_support << "\n";
                amazing << "reason=repeated high-lag family survived bias filter\n\n";
                amazing.flush();
            }
        }

        hits_file.flush();

        checkpoint_file << now_string()
                        << " epoch=" << epoch
                        << " offset=" << offset
                        << " hits=" << hits.size()
                        << " best_abs_z=" << best_abs_z
                        << " best_lag=" << best_lag
                        << " best_offset=" << best_offset
                        << " ones_z=" << ones_z
                        << " skipped_biased=" << skipped_biased
                        << "\n";
        checkpoint_file.flush();

        std::cout << "epoch " << epoch
                  << " | offset " << offset
                  << " | hits " << hits.size()
                  << " | ones_z " << ones_z
                  << " | best_abs_z " << best_abs_z
                  << " at lag " << best_lag
                  << "\n";

        if (!hits.empty()) {
            std::cout << "  strongest family hit: lag "
                      << hits[0].lag
                      << " z=" << hits[0].z
                      << " ratio=" << hits[0].ratio
                      << " repeats=" << hits[0].repeats
                      << " family_support=" << hits[0].family_support
                      << "\n";
        }
    }

    cudaFree(d_data);
    cudaFree(d_ones_blocks);
    cudaFree(d_partial);

    return 0;
}