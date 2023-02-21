import torch

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024 * 1024)


def reset_mem_usage():
    """
    Reset the GPU memory usage for the current device (GB).
    """
    torch.cuda.reset_peak_memory_stats()