"""
Task 41.2: Cache Coherence & Data Alignment Engine
Zero-Copy Memory Reshaper & SIMD Stride Enforcer
"""

import numpy as np

class CacheStrideAligner:
    """
    Enforces strict 64-byte physical memory alignment boundaries and SIMD register
    multiples for complex64 I/Q streams entering the DSP pipeline.
    Operates strictly via zero-copy NumPy memory views and address pointer arithmetic.
    """
    def __init__(self, channels=4, cache_line_bytes=64, element_bytes=8):
        self.channels = channels
        self.cache_line_bytes = cache_line_bytes
        self.element_bytes = element_bytes
        
        # 8 complex64 elements per 64-byte cache line
        self.elements_per_line = self.cache_line_bytes // self.element_bytes

    def generate_aligned_view(self, raw_buffer, expected_stride_len):
        """
        Dynamically extracts a strictly 64-byte aligned, SIMD-friendly continuous 
        planar representation from an over-provisioned raw memory buffer.
        
        Args:
            raw_buffer: 1D flat NumPy array (pre-allocated with padding).
            expected_stride_len: Desired number of samples per channel.
            
        Returns:
            aligned_planar_view: Zero-copy 2D array view (channels, SIMD-aligned stride_len)
            alignment_offset_bytes: The number of padding bytes bypassed.
            simd_stride_len: The exact executed stride length (truncated to SIMD multiple).
        """
        if raw_buffer.dtype != np.complex64:
            raise TypeError("Cache stride aligner requires strictly complex64 raw buffers.")

        # 1. Extract raw physical memory address pointer from ctypes buffer
        physical_address = raw_buffer.ctypes.data
        
        # 2. Compute byte offset to the nearest 64-byte boundary natively using bitwise masks
        # Offset = (64 - (address & 63)) & 63
        alignment_mask = self.cache_line_bytes - 1
        misalignment = physical_address & alignment_mask
        offset_bytes = (self.cache_line_bytes - misalignment) & alignment_mask
        
        # 3. Convert memory byte padding to discrete element offset
        offset_elements = offset_bytes // self.element_bytes
        
        # 4. Enforce strict SIMD vector execution width bounds on stride length
        # Bitwise truncation to guarantee length is exactly divisible by cache line elements
        simd_mask = ~(self.elements_per_line - 1)
        simd_stride_len = expected_stride_len & simd_mask
        
        # 5. Extract memory payload total (Channels * Stride)
        total_elements = self.channels * simd_stride_len
        
        # Protect against buffer overflow if not enough over-provisioned padding was supplied
        if offset_elements + total_elements > raw_buffer.size:
            raise BufferError(f"Raw buffer underflow. Requires {offset_elements + total_elements} elements, "
                              f"but received only {raw_buffer.size}. Ensure over-provisioning.")

        # 6. Extract zero-copy flat continuous cache-aligned slice
        aligned_flat_view = raw_buffer[offset_elements : offset_elements + total_elements]
        
        # 7. Reshape into 2D Planar structure (channels, stride_len)
        # Bypasses Python garbage collection / allocation by returning an NDArray view
        aligned_planar_view = aligned_flat_view.reshape((self.channels, simd_stride_len))
        
        return aligned_planar_view, offset_bytes, simd_stride_len

    def preallocate_aligned_buffer(self, expected_stride_len):
        """
        Helper utility to programmatically allocate an over-provisioned flat buffer 
        capable of guaranteeing a 64-byte boundary extraction view.
        """
        # Over-provision by 1 full cache line to guarantee alignment padding headroom
        simd_mask = ~(self.elements_per_line - 1)
        simd_stride_len = expected_stride_len & simd_mask
        
        total_elements = self.channels * simd_stride_len
        padded_size = total_elements + self.elements_per_line
        
        # Initial contiguous memory allocation
        raw_buffer = np.zeros(padded_size, dtype=np.complex64)
        
        # Return perfectly aligned view directly into the buffer
        aligned_view, offset, exact_stride = self.generate_aligned_view(raw_buffer, expected_stride_len)
        return raw_buffer, aligned_view, offset, exact_stride

if __name__ == "__main__":
    # Integration smoke test
    print("===================================================================")
    print("SPACESHIELD CACHE ALIGNMENT & SIMD STRIDE ENGINE")
    print("===================================================================")
    
    aligner = CacheStrideAligner(channels=4, cache_line_bytes=64, element_bytes=8)
    
    # Request 4096 elements
    raw_mem, planar_view, padding_offset, stride = aligner.preallocate_aligned_buffer(4096)
    
    ptr_address = planar_view.ctypes.data
    is_64b_aligned = (ptr_address & 63) == 0
    
    print(f"[INFO] Requested Stride Length:  4096")
    print(f"[INFO] SIMD Masked Stride Len:   {stride}")
    print(f"[INFO] Raw Buffer Payload Size:  {raw_mem.nbytes} bytes")
    print(f"[INFO] Bitwise Padding Applied:  {padding_offset} bytes")
    print(f"[INFO] Aligned Address Pointer:  0x{ptr_address:016X}")
    print(f"[INFO] 64-Byte Cache Alignment:  {'STRICT BOUNDARY CONFIRMED' if is_64b_aligned else 'FAILED'}")
    print(f"[INFO] 2D Array Memory Shape:    {planar_view.shape}")
    print(f"[INFO] Memory View Contiguous:   {planar_view.flags['C_CONTIGUOUS']}")
    
    assert is_64b_aligned, "Fatal Error: Cache memory is not 64-byte aligned."
    assert planar_view.flags['C_CONTIGUOUS'], "Fatal Error: Memory view is not C-contiguous."
    print("===================================================================")
    print("ALIGNMENT VERIFICATION COMPLETE.")
    print("===================================================================")
