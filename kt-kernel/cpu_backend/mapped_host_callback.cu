#include <stdio.h>
#include "mapped_host_callback.h"
#include <nvtx3/nvToolsExt.h>

#ifndef cudaErrorNotInitialized
#define cudaErrorNotInitialized ((cudaError_t)1000)  // Arbitrary unused error code
#endif

// ~100s avoiding GPU hang when main stream exits before callback completes
#define MAX_POLLS 100 * 1000 * 1000

// GPU kernel to signal callback readiness
__global__ void signal_callback_kernel(MappedHostCallback::CallbackData* data, void (*callback_func_ptr)(void*) , void* user_data_ptr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid == 0) {
        data->callback_func = callback_func_ptr;
        data->user_data = user_data_ptr;
        data->callback_completed=0;
        __threadfence_system(); 
        data->gpu_signal = 1;
        __threadfence_system();  // Ensure write is visible to CPU
    }
}

// GPU kernel to wait for callback completion (GPU-side synchronization)
__global__ void wait_callback_completion_kernel(MappedHostCallback::CallbackData* data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only thread 0 performs the polling
    if (tid == 0) {     // TODO thread management
        int poll_count = 0;
        while (data->callback_completed == 0 && poll_count < MAX_POLLS) {
            __threadfence_system();  // Ensure fresh read from CPU
            poll_count++;
        }
        if (poll_count == MAX_POLLS) {
            // return error to host side
            data->callback_completed = -1;  // Indicate timeout error
            __threadfence_system();
        }
    }
}

MappedHostCallback::MappedHostCallback() 
    : host_data(nullptr), device_data(nullptr), shutdown(false), initialized(false) {
    cudaError_t init_result = initialize();

    if (init_result == cudaSuccess) {
        initialized = true;
        // Start CPU polling thread
        polling_thread = std::thread(&MappedHostCallback::cpu_polling_loop, this);  // TODO 
    } else {
        printf("[MappedHostCallback] ERROR: Failed to initialize MappedHostCallback: %s\n", cudaGetErrorString(init_result));
    }
}

MappedHostCallback::~MappedHostCallback() {
    if (initialized) {
        // Signal shutdown
        shutdown.store(true);
        
        // Wait for polling thread to finish
        if (polling_thread.joinable()) {
            polling_thread.join();
        }
        
        cleanup();
    }
}

cudaError_t MappedHostCallback::initialize() {
    // Allocate mapped host memory
    cudaError_t err = cudaHostAlloc((void**)&host_data, sizeof(CallbackData), cudaHostAllocMapped);
    if (err != cudaSuccess) {
        printf("cudaHostAlloc failed! Error: %s\n", cudaGetErrorString(err));
        return err;
    }

    // Get device pointer for the mapped memory
    err = cudaHostGetDevicePointer((void**)&device_data, (void*)host_data, 0);
    if (err != cudaSuccess) {
        // print error and clean up

        printf("cudaHostGetDevicePointer failed! Error: %s\n", cudaGetErrorString(err));
        cudaFreeHost(host_data);
        host_data = nullptr;
        return err;
    }
    
    // Check if the device supports mapped memory
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (!prop.canMapHostMemory) {
        printf("Device does not support mapped host memory!\n");
        cudaFreeHost(host_data);
        host_data = nullptr;
        return cudaErrorNotSupported;
    }

    // Initialize the data structure
    host_data->gpu_signal = 0;
    host_data->callback_func = nullptr;
    host_data->user_data = nullptr;
    host_data->callback_completed = 1;  // No callback pending
    
    return cudaSuccess;
}

void MappedHostCallback::cleanup() {
    if (host_data) {
        cudaFreeHost(host_data);
        host_data = nullptr;
        device_data = nullptr;
    }
}

cudaError_t MappedHostCallback::launch_host_func(cudaStream_t stream, void (*callback)(void*), void* data) {
    if (!initialized) {
        printf("ERROR: MappedHostCallback not initialized!\n");
        return cudaErrorNotInitialized;
    }

    // Check if device_data is valid
    if (!device_data) {
        printf("ERROR: device_data is null!\n");
        return cudaErrorInvalidValue;
    }
    
    // Launch signal kernel on the stream
    signal_callback_kernel<<<1, 1, 0, stream>>>(device_data, callback, data);
    
    // Check for kernel launch error
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        printf("ERROR: Kernel launch failed! Error: %s\n", cudaGetErrorString(launch_err));
    }
    return launch_err;
}

cudaError_t MappedHostCallback::synchronize_on_gpu(cudaStream_t stream) {
    if (!initialized) {
        return cudaErrorNotInitialized;
    }
    
    wait_callback_completion_kernel<<<1, 1, 0, stream>>>(device_data);
    return cudaGetLastError();
}

void MappedHostCallback::cpu_polling_loop() {
    printf("Starting polling, gpu_signal:%d, callback_completed:%d\n", host_data->gpu_signal, host_data->callback_completed);
    while (!shutdown.load()) {
        // Check if GPU has signaled and user data is ready
        if (host_data->gpu_signal == 1 && host_data->callback_func != nullptr && host_data->user_data != nullptr) {
            host_data->callback_func(host_data->user_data);

            // Reset for next callback
            host_data->callback_func = nullptr;
            host_data->user_data = nullptr;
            host_data->gpu_signal = 0;

            // Mark callback as completed
            if (host_data->callback_completed == -1) {
                printf("ERROR: wait callback completion timed out on GPU side!\n");
                break;
            }
            host_data->callback_completed = 1;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        
        // Small sleep to avoid busy waiting
        //std::this_thread::sleep_for(std::chrono::microseconds(1));
    }
    printf("Exiting polling loop\n");
}