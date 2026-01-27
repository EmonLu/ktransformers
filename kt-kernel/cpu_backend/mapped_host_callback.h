#include <cuda_runtime.h>
#include <atomic>
#include <thread>

class MappedHostCallback {
public:
    struct CallbackData {
        volatile int32_t gpu_signal;      // GPU writes here when ready (0=waiting, 1=ready)
        void (*callback_func)(void*);         // Function to call
        void* user_data;                      // User data pointer (guaranteed valid)
        volatile int32_t callback_completed; // Callback completion flag
    };
    
    CallbackData* host_data;                  // Host-accessible data
    CallbackData* device_data;                // Device-accessible data
    std::thread polling_thread;               // CPU polling thread
    bool initialized;                         // Initialization flag
    std::atomic<bool> shutdown;               // Shutdown flag

public:
    MappedHostCallback();
    ~MappedHostCallback();
    // Replace cudaLaunchHostFunc functionality
    cudaError_t launch_host_func(cudaStream_t stream, void (*callback)(void*), void* data);
    
    // Launch GPU kernel to wait for callback completion (GPU-side sync)
    cudaError_t synchronize_on_gpu(cudaStream_t stream);

private:
    // Initialize mapped memory
    cudaError_t initialize();
    
    // Cleanup mapped memory
    void cleanup();
    
    // CPU polling loop
    void cpu_polling_loop();
};