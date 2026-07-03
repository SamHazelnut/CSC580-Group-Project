#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <string>
#include <mpi.h>

// Structure to hold all analytical results
struct Results {
    double mean;
    double variance;
    double stddev;
    double min_val;
    double max_val;
    std::vector<int> histogram;
    double correlation;
    std::vector<double> moving_average;
    std::vector<double> outliers;
    
    // Per-task timing (in milliseconds)
    double time_basic_stats;
    double time_histogram;
    double time_sort;
    double time_correlation;
    double time_moving_average;
    double time_outliers;
    double total_time;
};

class MPIAnalytics {
private:
    int rank, world_size;
    size_t data_size;
    std::vector<double> full_data;     // Only on rank 0
    std::vector<double> full_data2;    // Only on rank 0 (for correlation)
    std::vector<double> local_data;    // Each node's chunk
    Results results;
    
public:
    MPIAnalytics() {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    }
    
    ~MPIAnalytics() {
        MPI_Finalize();
    }
    
    // Getter for rank
    int getRank() const {
        return rank;
    }
    
    // Generate dataset on master only
    void generateData(size_t n) {
        data_size = n;
        
        if (rank == 0) {
            full_data.resize(data_size);
            full_data2.resize(data_size);
            
            std::mt19937_64 rng(42);
            std::uniform_real_distribution<double> dist(0.0, 10000.0);
            
            for (size_t i = 0; i < data_size; ++i) {
                full_data[i] = dist(rng);
                full_data2[i] = dist(rng);
            }
            
            std::cout << "Master: Generated " << data_size << " data points\n";
        }
    }
    
    // Distribute data to all nodes using MPI_Scatterv
    void distributeData() {
        // Broadcast data size to all nodes
        MPI_Bcast(&data_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        // Calculate chunk sizes for each node
        size_t base_chunk = data_size / world_size;
        size_t remainder = data_size % world_size;
        
        std::vector<int> send_counts(world_size);
        std::vector<int> displacements(world_size);
        
        for (int i = 0; i < world_size; ++i) {
            send_counts[i] = base_chunk + (i < (int)remainder ? 1 : 0);
            displacements[i] = (i == 0) ? 0 : displacements[i-1] + send_counts[i-1];
        }
        
        int local_count = send_counts[rank];
        local_data.resize(local_count);
        
        // Scatter data
        MPI_Scatterv(
            (rank == 0) ? full_data.data() : nullptr,
            send_counts.data(),
            displacements.data(),
            MPI_DOUBLE,
            local_data.data(),
            local_count,
            MPI_DOUBLE,
            0,
            MPI_COMM_WORLD
        );
        
        if (rank == 0) {
            std::cout << "Data distributed: " << local_count << " points per node\n";
        }
    }
    
    // Task 1: Basic Statistics (Parallel)
    void computeBasicStats() {
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        // Local computation
        double local_sum = 0.0;
        double local_sum_sq = 0.0;
        double local_min = std::numeric_limits<double>::max();
        double local_max = std::numeric_limits<double>::lowest();
        
        for (double val : local_data) {
            local_sum += val;
            local_sum_sq += val * val;
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
        }
        
        // Global reduction
        double global_sum, global_sum_sq, global_min, global_max;
        MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sum_sq, &global_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            results.mean = global_sum / data_size;
            results.variance = (global_sum_sq / data_size) - (results.mean * results.mean);
            results.stddev = std::sqrt(results.variance);
            results.min_val = global_min;
            results.max_val = global_max;
            
            double end_time = MPI_Wtime();
            results.time_basic_stats = (end_time - start_time) * 1000.0;
            std::cout << "✓ Basic Statistics: " << results.time_basic_stats << " ms\n";
        }
    }
    
    // Task 2: Histogram (Parallel)
    void computeHistogram(int num_bins = 20) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        // Broadcast min/max from master
        double global_min = 0.0, global_max = 0.0;
        if (rank == 0) {
            global_min = results.min_val;
            global_max = results.max_val;
        }
        MPI_Bcast(&global_min, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_max, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Local histogram
        std::vector<int> local_hist(num_bins, 0);
        double range = global_max - global_min;
        double bin_width = range / num_bins;
        
        for (double val : local_data) {
            int bin = static_cast<int>((val - global_min) / bin_width);
            if (bin == num_bins) bin = num_bins - 1;
            local_hist[bin]++;
        }
        
        // Reduce histograms
        std::vector<int> global_hist(num_bins, 0);
        MPI_Reduce(local_hist.data(), global_hist.data(), num_bins, 
                   MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            results.histogram = global_hist;
            double end_time = MPI_Wtime();
            results.time_histogram = (end_time - start_time) * 1000.0;
            std::cout << "✓ Histogram: " << results.time_histogram << " ms\n";
        }
    }
    
    // Task 3: Parallel Sort
    void computeSort() {
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        // Local sort
        std::sort(local_data.begin(), local_data.end());
        
        // Gather all sorted data to master
        int local_size = local_data.size();
        std::vector<int> recv_counts(world_size);
        MPI_Gather(&local_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<int> displacements(world_size);
        std::vector<double> global_sorted;
        
        if (rank == 0) {
            global_sorted.resize(data_size);
            for (int i = 0; i < world_size; ++i) {
                displacements[i] = (i == 0) ? 0 : displacements[i-1] + recv_counts[i-1];
            }
        }
        
        MPI_Gatherv(local_data.data(), local_size, MPI_DOUBLE,
                   global_sorted.data(), recv_counts.data(), 
                   displacements.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            // Final merge sort on master
            std::sort(global_sorted.begin(), global_sorted.end());
            double end_time = MPI_Wtime();
            results.time_sort = (end_time - start_time) * 1000.0;
            std::cout << "✓ Sorting: " << results.time_sort << " ms\n";
        }
    }
    
    // Task 4: Correlation (Parallel)
    void computeCorrelation() {
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        // Generate second column locally with same seed
        std::mt19937_64 rng(42 + rank);
        std::uniform_real_distribution<double> dist(0.0, 10000.0);
        
        double local_sum_x = 0.0, local_sum_y = 0.0;
        double local_sum_xy = 0.0, local_sum_x2 = 0.0, local_sum_y2 = 0.0;
        
        for (double x : local_data) {
            double y = dist(rng);
            local_sum_x += x;
            local_sum_y += y;
            local_sum_xy += x * y;
            local_sum_x2 += x * x;
            local_sum_y2 += y * y;
        }
        
        // Global reduction
        double sum_x, sum_y, sum_xy, sum_x2, sum_y2;
        MPI_Reduce(&local_sum_x, &sum_x, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sum_y, &sum_y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sum_xy, &sum_xy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sum_x2, &sum_x2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sum_y2, &sum_y2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            double numerator = data_size * sum_xy - sum_x * sum_y;
            double denominator = std::sqrt((data_size * sum_x2 - sum_x * sum_x) * 
                                          (data_size * sum_y2 - sum_y * sum_y));
            results.correlation = numerator / denominator;
            
            double end_time = MPI_Wtime();
            results.time_correlation = (end_time - start_time) * 1000.0;
            std::cout << "✓ Correlation: " << results.time_correlation << " ms\n";
        }
    }
    
    // Task 5: Moving Average (Parallel)
    void computeMovingAverage(int window_size = 1000) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        // Broadcast window size
        MPI_Bcast(&window_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        // Local moving average
        std::vector<double> local_ma;
        size_t local_n = local_data.size();
        
        if (local_n >= (size_t)window_size) {
            double window_sum = 0.0;
            for (int i = 0; i < window_size; ++i) {
                window_sum += local_data[i];
            }
            local_ma.push_back(window_sum / window_size);
            
            for (size_t i = window_size; i < local_n; ++i) {
                window_sum += local_data[i] - local_data[i - window_size];
                local_ma.push_back(window_sum / window_size);
            }
        }
        
        // Gather all moving averages
        int local_ma_size = local_ma.size();
        std::vector<int> recv_counts(world_size);
        MPI_Gather(&local_ma_size, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<int> displacements(world_size);
        std::vector<double> global_ma;
        
        if (rank == 0) {
            int total_ma_size = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
            global_ma.resize(total_ma_size);
            
            for (int i = 0; i < world_size; ++i) {
                displacements[i] = (i == 0) ? 0 : displacements[i-1] + recv_counts[i-1];
            }
        }
        
        MPI_Gatherv(local_ma.data(), local_ma_size, MPI_DOUBLE,
                   global_ma.data(), recv_counts.data(), 
                   displacements.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            results.moving_average = global_ma;
            double end_time = MPI_Wtime();
            results.time_moving_average = (end_time - start_time) * 1000.0;
            std::cout << "✓ Moving Average: " << results.time_moving_average << " ms\n";
        }
    }
    
    // Task 6: Outlier Detection (Parallel)
    void detectOutliers(double threshold = 3.0) {
        MPI_Barrier(MPI_COMM_WORLD);
        double start_time = MPI_Wtime();
        
        // Broadcast mean and stddev
        double global_mean = 0.0, global_stddev = 0.0;
        if (rank == 0) {
            global_mean = results.mean;
            global_stddev = results.stddev;
        }
        MPI_Bcast(&global_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_stddev, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&threshold, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        // Local outlier detection
        std::vector<double> local_outliers;
        for (double val : local_data) {
            double z_score = (val - global_mean) / global_stddev;
            if (std::abs(z_score) > threshold) {
                local_outliers.push_back(val);
            }
        }
        
        // Gather all outliers
        int local_count = local_outliers.size();
        std::vector<int> recv_counts(world_size);
        MPI_Gather(&local_count, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<int> displacements(world_size);
        std::vector<double> global_outliers;
        
        if (rank == 0) {
            int total_outliers = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
            global_outliers.resize(total_outliers);
            
            for (int i = 0; i < world_size; ++i) {
                displacements[i] = (i == 0) ? 0 : displacements[i-1] + recv_counts[i-1];
            }
        }
        
        MPI_Gatherv(local_outliers.data(), local_count, MPI_DOUBLE,
                   global_outliers.data(), recv_counts.data(), 
                   displacements.data(), MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        
        if (rank == 0) {
            results.outliers = global_outliers;
            double end_time = MPI_Wtime();
            results.time_outliers = (end_time - start_time) * 1000.0;
            std::cout << "✓ Outlier Detection: " << results.time_outliers << " ms\n";
            std::cout << "  Found " << global_outliers.size() << " outliers\n";
        }
    }
    
    // Run all analytics
    void runAll(size_t n) {
        double total_start = MPI_Wtime();
        
        generateData(n);
        distributeData();
        
        if (rank == 0) {
            std::cout << "\n=== MPI Parallel Analytics Running ===\n";
            std::cout << "Dataset size: " << data_size << "\n";
            std::cout << "Nodes: " << world_size << "\n\n";
        }
        
        computeBasicStats();
        computeHistogram();
        computeSort();
        computeCorrelation();
        computeMovingAverage();
        detectOutliers();
        
        if (rank == 0) {
            double total_end = MPI_Wtime();
            results.total_time = (total_end - total_start) * 1000.0;
            std::cout << "\n✓ Total execution time: " << results.total_time << " ms\n";
            std::cout << "✓ Total: " << results.total_time / 1000.0 << " seconds\n";
        }
    }
    
    // Save results to CSV
    void saveResults(const std::string& filename) {
        if (rank == 0) {
            std::ofstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Could not open " << filename << "\n";
                return;
            }
            
            file << "Metric,Value\n";
            file << "Mean," << results.mean << "\n";
            file << "Variance," << results.variance << "\n";
            file << "StdDev," << results.stddev << "\n";
            file << "Min," << results.min_val << "\n";
            file << "Max," << results.max_val << "\n";
            file << "Correlation," << results.correlation << "\n";
            file << "OutlierCount," << results.outliers.size() << "\n";
            file << "BasicStatsTime_ms," << results.time_basic_stats << "\n";
            file << "HistogramTime_ms," << results.time_histogram << "\n";
            file << "SortTime_ms," << results.time_sort << "\n";
            file << "CorrelationTime_ms," << results.time_correlation << "\n";
            file << "MovingAverageTime_ms," << results.time_moving_average << "\n";
            file << "OutlierTime_ms," << results.time_outliers << "\n";
            file << "TotalTime_ms," << results.total_time << "\n";
            
            file.close();
            std::cout << "Results saved to " << filename << "\n";
        }
    }
    
    // Print summary
    void printSummary() {
        if (rank == 0) {
            std::cout << "\n=== Results Summary ===\n";
            std::cout << "Mean: " << results.mean << "\n";
            std::cout << "Variance: " << results.variance << "\n";
            std::cout << "StdDev: " << results.stddev << "\n";
            std::cout << "Min: " << results.min_val << "\n";
            std::cout << "Max: " << results.max_val << "\n";
            std::cout << "Correlation: " << results.correlation << "\n";
            std::cout << "Outliers: " << results.outliers.size() << "\n";
        }
    }
};

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Parse command line argument
    size_t data_size = 10000000;  // Default: 10 million
    
    if (argc > 1) {
        data_size = std::stoull(argv[1]);
    }
    
    // Create MPI analytics object
    MPIAnalytics analytics;
    
    // Print header from rank 0 only
    if (analytics.getRank() == 0) {
        std::cout << "MPI Parallel Analytics\n";
        std::cout << "Data size: " << data_size << "\n";
    }
    
    analytics.runAll(data_size);
    analytics.printSummary();
    analytics.saveResults("mpi_results.csv");
    
    return 0;
}