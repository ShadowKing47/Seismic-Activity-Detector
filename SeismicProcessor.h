#ifndef SEISMIC_PROCESSOR_H
#define SEISMIC_PROCESSOR_H

#include <vector>
#include <array>
#include <cstdint>
#include <memory>

// Constants from Python implementation
constexpr int WINDOW_SIZE = 128;
constexpr int HOP_LENGTH = 64;
constexpr int SAMPLE_RATE = 1000;
constexpr int TARGET_SAMPLE_RATE = 100;
constexpr int DECIMATION_FACTOR = SAMPLE_RATE / TARGET_SAMPLE_RATE;
constexpr int FIR_TAPS = 15;

class FIRFilter {
public:
    FIRFilter();
    float processSample(float sample);
    void processBuffer(const std::vector<float>& input, std::vector<float>& output);
    
private:
    std::array<float, FIR_TAPS> buffer_;
    std::array<float, FIR_TAPS> coefficients_;
    size_t bufferIndex_ = 0;
};

class SeismicProcessor {
public:
    SeismicProcessor();
    
    // Process a single sample through the pipeline
    bool processSample(float sample);
    
    // Process a buffer of samples
    void processBuffer(const std::vector<float>& input, std::vector<float>& output);
    
    // Get the current window for ML processing
    const std::vector<float>& getCurrentWindow() const { return windowBuffer_; }
    
    // Reset the processor state
    void reset();
    
private:
    FIRFilter firFilter_;
    std::vector<float> windowBuffer_;
    size_t samplesSinceLastInference_ = 0;
    
    // ML model output buffer
    std::vector<float> mlOutput_;
    
    // Internal state for downsampling
    size_t sampleCounter_ = 0;
    
    // Process the current window with ML model
    void processWindow();
};

#endif // SEISMIC_PROCESSOR_H
