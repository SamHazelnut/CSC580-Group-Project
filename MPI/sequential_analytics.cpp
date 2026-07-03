#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <string>

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

class SequentialAnalytics {
private:
    std::vector<double> data;      // Column 1
    std::vector<double> data2;     // Column 2 (for correlation)
    Results results;
    size_t data_size;
    
public:
    SequentialAnalytics(size_t n) : data_size(n) {
        generateData();
    }
    
    // Generate dataset with fixed seed for reproducibility
    void generateData() {
        data.resize(data_size);
        data2.resize(data_size);
        
        std::mt19937_64 rng(42);  // Fixed seed
        std::uniform_real_distribution<double> dist(0.0, 10000.0);
        
        for (size_t i = 0; i < data_size; ++i) {
            data[i] = dist(rng);
            data2[i] = dist(rng);  // Independent second column
        }
    }
    
    // Task 1: Basic Statistics (Mean, Variance, StdDev, Min, Max)
    void computeBasicStats() {
        auto start = std::chrono::high_resolution_clock::now();
        
        double sum = 0.0, sum_sq = 0.0;
        results.min_val = std::numeric_limits<double>::max();
        results.max_val = std::numeric_limits<double>::lowest();
        
        for (double val : data) {
            sum += val;
            sum_sq += val * val;
            if (val < results.min_val) results.min_val = val;
            if (val > results.max_val) results.max_val = val;
        }
        
        results.mean = sum / data_size;
        results.variance = (sum_sq / data_size) - (results.mean * results.mean);
        results.stddev = std::sqrt(results.variance);
        
        auto end = std::chrono::high_resolution_clock::now();
        results.time_basic_stats = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Task 2: Histogram Generation (20 bins)
    void computeHistogram(int num_bins = 20) {
        auto start = std::chrono::high_resolution_clock::now();
        
        results.histogram.assign(num_bins, 0);
        double range = results.max_val - results.min_val;
        double bin_width = range / num_bins;
        
        for (double val : data) {
            int bin = static_cast<int>((val - results.min_val) / bin_width);
            if (bin == num_bins) bin = num_bins - 1;
            results.histogram[bin]++;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        results.time_histogram = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Task 3: Sorting
    void computeSort() {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<double> sorted = data;
        std::sort(sorted.begin(), sorted.end());
        
        auto end = std::chrono::high_resolution_clock::now();
        results.time_sort = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Task 4: Pearson Correlation
    void computeCorrelation() {
        auto start = std::chrono::high_resolution_clock::now();
        
        double sum_x = 0.0, sum_y = 0.0;
        double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
        
        for (size_t i = 0; i < data_size; ++i) {
            double x = data[i];
            double y = data2[i];
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
        }
        
        double numerator = data_size * sum_xy - sum_x * sum_y;
        double denominator = std::sqrt((data_size * sum_x2 - sum_x * sum_x) * 
                                      (data_size * sum_y2 - sum_y * sum_y));
        results.correlation = numerator / denominator;
        
        auto end = std::chrono::high_resolution_clock::now();
        results.time_correlation = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Task 5: Moving Average (window size = 1000)
    void computeMovingAverage(int window_size = 1000) {
        auto start = std::chrono::high_resolution_clock::now();
        
        results.moving_average.clear();
        if (data_size < window_size) {
            results.moving_average.push_back(results.mean);
            auto end = std::chrono::high_resolution_clock::now();
            results.time_moving_average = std::chrono::duration<double, std::milli>(end - start).count();
            return;
        }
        
        results.moving_average.reserve(data_size - window_size + 1);
        
        // Initial window sum
        double window_sum = 0.0;
        for (int i = 0; i < window_size; ++i) {
            window_sum += data[i];
        }
        results.moving_average.push_back(window_sum / window_size);
        
        // Slide window
        for (size_t i = window_size; i < data_size; ++i) {
            window_sum += data[i] - data[i - window_size];
            results.moving_average.push_back(window_sum / window_size);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        results.time_moving_average = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Task 6: Outlier Detection (Z-score method, |Z| > 3)
    void detectOutliers(double threshold = 3.0) {
        auto start = std::chrono::high_resolution_clock::now();
        
        results.outliers.clear();
        for (double val : data) {
            double z_score = (val - results.mean) / results.stddev;
            if (std::abs(z_score) > threshold) {
                results.outliers.push_back(val);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        results.time_outliers = std::chrono::duration<double, std::milli>(end - start).count();
    }
    
    // Run all analytics
    void runAll() {
        std::cout << "\n=== Sequential Analytics Running ===\n";
        std::cout << "Dataset size: " << data_size << "\n\n";
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        computeBasicStats();
        std::cout << "✓ Basic Statistics: " << results.time_basic_stats << " ms\n";
        
        computeHistogram();
        std::cout << "✓ Histogram: " << results.time_histogram << " ms\n";
        
        computeSort();
        std::cout << "✓ Sorting: " << results.time_sort << " ms\n";
        
        computeCorrelation();
        std::cout << "✓ Correlation: " << results.time_correlation << " ms\n";
        
        computeMovingAverage();
        std::cout << "✓ Moving Average: " << results.time_moving_average << " ms\n";
        
        detectOutliers();
        std::cout << "✓ Outlier Detection: " << results.time_outliers << " ms\n";
        std::cout << "  Found " << results.outliers.size() << " outliers\n";
        
        auto total_end = std::chrono::high_resolution_clock::now();
        results.total_time = std::chrono::duration<double, std::milli>(total_end - total_start).count();
        
        std::cout << "\n✓ Total execution time: " << results.total_time << " ms\n";
        std::cout << "✓ Total: " << results.total_time / 1000.0 << " seconds\n";
    }
    
    // Save results to CSV
    void saveResults(const std::string& filename) {
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
    
    // Print summary
    void printSummary() {
        std::cout << "\n=== Results Summary ===\n";
        std::cout << "Mean: " << results.mean << "\n";
        std::cout << "Variance: " << results.variance << "\n";
        std::cout << "StdDev: " << results.stddev << "\n";
        std::cout << "Min: " << results.min_val << "\n";
        std::cout << "Max: " << results.max_val << "\n";
        std::cout << "Correlation: " << results.correlation << "\n";
        std::cout << "Outliers: " << results.outliers.size() << "\n";
    }
};

int main(int argc, char* argv[]) {
    // Parse command line argument
    size_t data_size = 10000000;  // Default: 10 million
    
    if (argc > 1) {
        data_size = std::stoull(argv[1]);
    }
    
    std::cout << "Sequential Analytics\n";
    std::cout << "Data size: " << data_size << "\n";
    
    SequentialAnalytics analytics(data_size);
    analytics.runAll();
    analytics.printSummary();
    analytics.saveResults("sequential_results.csv");
    
    return 0;
}