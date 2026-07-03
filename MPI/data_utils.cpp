#include <vector>
#include <random>
#include <iostream>
#include <cmath>

// ============================================================
// FUNCTION 1: Generate synthetic data
// ============================================================
std::vector<double> generate_data(size_t N) {
    std::cout << "==================================================" << std::endl;
    std::cout << "  DATA ENGINEER (Member 3) GENERATING DATASET" << std::endl;
    std::cout << "  Dataset size: " << N << " data points" << std::endl;
    std::cout << "  Fixed seed: 42 (reproducible)" << std::endl;
    std::cout << "  Range: [0, 10000]" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    std::vector<double> data(N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 10000.0);
    
    for (size_t i = 0; i < N; ++i) {
        data[i] = dist(rng);
    }
    
    return data;
}

// ============================================================
// FUNCTION 2: Calculate how to split data across nodes
// ============================================================
void calculate_scatterv_params(int total_N, int world_size, 
                               std::vector<int>& sendcounts, 
                               std::vector<int>& displacements) {
    std::cout << "[Data Engineer] Partitioning data across " 
              << world_size << " nodes" << std::endl;
    
    sendcounts.resize(world_size);
    displacements.resize(world_size);

    int base_count = total_N / world_size;
    int remainder = total_N % world_size;

    int sum = 0;
    for (int i = 0; i < world_size; i++) {
        sendcounts[i] = (i < remainder) ? base_count + 1 : base_count;
        displacements[i] = sum;
        sum += sendcounts[i];
        
        std::cout << "  Node " << i << ": " << sendcounts[i] 
                  << " items, starting at position " << displacements[i] << std::endl;
    }
}

// ============================================================
// FUNCTION 3: Results structure
// ============================================================
struct Results {
    double sum;
    double mean;
    double variance;
    double min;
    double max;
    long long count;
};

// ============================================================
// FUNCTION 4: Compute statistics on a chunk of data
// ============================================================
Results compute_results(const std::vector<double>& data) {
    Results res;
    res.count = data.size();
    res.sum = 0.0;
    res.min = 1e9;
    res.max = -1e9;
    
    for (double val : data) {
        res.sum += val;
        if (val < res.min) res.min = val;
        if (val > res.max) res.max = val;
    }
    res.mean = res.sum / res.count;
    
    double sq_sum = 0.0;
    for (double val : data) {
        sq_sum += (val - res.mean) * (val - res.mean);
    }
    res.variance = sq_sum / res.count;
    
    return res;
}

// ============================================================
// FUNCTION 5: Validate MPI results match sequential results
// ============================================================
bool validate_results(const Results& seq, const Results& mpi, double tolerance = 1e-6) {
    std::cout << "[Data Engineer] Validating results..." << std::endl;
    
    bool all_match = true;
    
    if (std::abs(seq.mean - mpi.mean) > tolerance) {
        std::cout << "  ✗ Mean mismatch: " << seq.mean << " vs " << mpi.mean << std::endl;
        all_match = false;
    } else {
        std::cout << "  ✓ Mean matches" << std::endl;
    }
    
    if (std::abs(seq.variance - mpi.variance) > tolerance) {
        std::cout << "  ✗ Variance mismatch" << std::endl;
        all_match = false;
    } else {
        std::cout << "  ✓ Variance matches" << std::endl;
    }
    
    if (std::abs(seq.min - mpi.min) > tolerance) {
        std::cout << "  ✗ Min mismatch" << std::endl;
        all_match = false;
    } else {
        std::cout << "  ✓ Min matches" << std::endl;
    }
    
    if (std::abs(seq.max - mpi.max) > tolerance) {
        std::cout << "  ✗ Max mismatch" << std::endl;
        all_match = false;
    } else {
        std::cout << "  ✓ Max matches" << std::endl;
    }
    
    if (std::abs(seq.sum - mpi.sum) > tolerance) {
        std::cout << "  ✗ Sum mismatch" << std::endl;
        all_match = false;
    } else {
        std::cout << "  ✓ Sum matches" << std::endl;
    }
    
    if (all_match) {
        std::cout << "✅ DATA VALIDATION PASSED - All results match!" << std::endl;
        std::cout << "   (Data Engineer's validation function confirms correctness)" << std::endl;
    } else {
        std::cout << "❌ DATA VALIDATION FAILED - Results don't match!" << std::endl;
    }
    
    return all_match;
}

// ============================================================
// NOTE: NO main() function here!
// The main() is in sequential_analytics.cpp or mpi_analytics.cpp
// ============================================================