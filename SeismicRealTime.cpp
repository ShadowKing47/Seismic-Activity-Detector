#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "libmseed.h"

// Constants from Python implementation
const int WINDOW_SIZE = 128;
const int HOP_LENGTH = 64;
const int SAMPLE_RATE = 1000;
const int TARGET_SAMPLE_RATE = 100;
const int DECIMATION_FACTOR = SAMPLE_RATE / TARGET_SATE_RATE;
const int FIR_TAPS = 15;

// FIR Filter implementation
class FIRFilter {
private:
    std::vector<double> buffer;
    std::vector<double> coefficients;
    size_t bufferIndex;
    
public:
    FIRFilter() : buffer(FIR_TAPS, 0.0), bufferIndex(0) {
        // Initialize with a simple low-pass filter
        coefficients = {
            -0.0087, 0.0119, 0.0279, 0.0496, 0.0733, 0.0951,
            0.1114, 0.1200, 0.1200, 0.1114, 0.0951, 0.0733,
            0.0496, 0.0279, 0.0119, -0.0087
        };
    }
    
    double processSample(double sample) {
        buffer[bufferIndex] = sample;
        double result = 0.0;
        
        // Apply FIR filter
        for (size_t i = 0; i < FIR_TAPS; ++i) {
            result += coefficients[i] * buffer[(bufferIndex + i) % FIR_TAPS];
        }
        
        bufferIndex = (bufferIndex + 1) % FIR_TAPS;
        return result;
    }
};

// Main processing class
class SeismicProcessor {
private:
    FIRFilter firFilter;
    std::vector<double> windowBuffer;
    size_t sampleCounter;
    
public:
    SeismicProcessor() : sampleCounter(0) {
        windowBuffer.reserve(WINDOW_SIZE);
    }
    
    void processSample(double sample) {
        // Apply FIR filter
        double filtered = firFilter.processSample(sample);
        
        // Downsample
        if (sampleCounter++ % DECIMATION_FACTOR != 0) {
            return;
        }
        
        // Add to window buffer
        windowBuffer.push_back(filtered);
        
        // Process complete window
        if (windowBuffer.size() >= WINDOW_SIZE) {
            processWindow();
            // Slide window by hop length
            windowBuffer.erase(windowBuffer.begin(), windowBuffer.begin() + HOP_LENGTH);
        }
    }
    
    void processWindow() {
        if (windowBuffer.size() < WINDOW_SIZE) return;
        
        // Here you would typically pass the window to your ML model
        // For now, we'll just print some statistics
        double sum = 0.0;
        double min_val = windowBuffer[0];
        double max_val = windowBuffer[0];
        
        for (double val : windowBuffer) {
            sum += val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
        
        double mean = sum / windowBuffer.size();
        
        std::cout << "Window processed - "
                  << "Min: " << min_val << ", "
                  << "Max: " << max_val << ", "
                  << "Mean: " << mean << std::endl;
    }
};

// Function to read MSEED file
void processMseedFile(const std::string& filename, SeismicProcessor& processor) {
    MS3Record *msr = NULL;
    uint32_t flags = MSF_VALIDATECRC;
    int8_t verbose = 0;
    
    // Open the file
    msr = msr3_init(msr);
    if (!msr) {
        std::cerr << "Error initializing MSRecord" << std::endl;
        return;
    }
    
    int ret = ms3_readmsr(&msr, filename.c_str(), NULL, NULL, flags, verbose);
    
    if (ret != MS_NOERROR) {
        std::cerr << "Error reading file: " << filename << std::endl;
        msr3_free(&msr);
        return;
    }
    
    // Process each sample
    for (int i = 0; i < msr->samplecnt; i++) {
        // Convert sample to double (assuming 32-bit float samples)
        double sample = msr->datasamples.f[i];
        processor.processSample(sample);
    }
    
    msr3_free(&msr);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <mseed_file>" << std::endl;
        return 1;
    }
    
    std::string filename = argv[1];
    SeismicProcessor processor;
    
    auto start = std::chrono::high_resolution_clock::now();
    processMseedFile(filename, processor);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Processing completed in " << duration.count() << " ms" << std::endl;
    
    return 0;
}
