"""
Task 70.2: Dynamic Overlap Window Manager Module
SpaceShield High-Velocity Receiver DSP Subsystem

Adjusts the overlap-save discard length and re-aligns sliding buffers
whenever the FFT block size is dynamically modified by the tuner.
"""

import numpy as np
from numba import njit

@njit(fastmath=True, cache=True, nopython=True, boundscheck=False)
def _adjust_overlap_buffer_jit(
    old_buf: np.ndarray,      # (64, 4) complex64
    old_fft_size: int,
    new_fft_size: int,
    new_buf: np.ndarray       # (64, 4) complex64 (output)
) -> int:
    """
    Zero-Heap JIT sliding buffer re-alignment during FFT block size transitions.
    Copies appropriate historical samples to the new buffer to prevent boundary jumps.
    Returns the new discard overlap length D.
    """
    new_D = new_fft_size // 2
    
    # Source index in old buffer to read from
    start_idx = old_fft_size - new_D
    
    # Clean output buffer
    new_buf.fill(0.0 + 0.0j)
    
    for t in range(new_D):
        # Prevent indexing overflow if the old size was very small
        src_t = max(0, start_idx + t)
        for c in range(4):
            new_buf[t, c] = old_buf[src_t, c]
            
    return new_D


class DynamicOverlapWindowManager:
    """
    Coordinates block boundary alignments during adaptive FFT size transitions.
    Preserves timing and phase continuity without memory allocations in the hot path.
    """
    def __init__(self):
        # Pre-allocated temporary sliding buffers (maximum size 64 x 4 elements)
        self.temp_buf = np.zeros((64, 4), dtype=np.complex64)
        self.active_fft_size = 32  # Default starting size
        self.active_discard = 16

    def transition_block_size(
        self,
        current_buffer: np.ndarray,  # (64, 4) active sliding buffer
        new_fft_size: int
    ) -> int:
        """
        Executes JIT re-alignment of the buffer.
        Modifies current_buffer in-place.
        Returns the new discard overlap length D.
        """
        # Run JIT alignment on temp buffer
        new_D = _adjust_overlap_buffer_jit(
            current_buffer,
            self.active_fft_size,
            new_fft_size,
            self.temp_buf
        )
        
        # Copy back to current_buffer in-place
        current_buffer.fill(0.0 + 0.0j)
        for t in range(new_fft_size):
            for c in range(4):
                current_buffer[t, c] = self.temp_buf[t, c]
                
        # Save active configuration states
        self.active_fft_size = new_fft_size
        self.active_discard = new_D
        
        return new_D


# =========================================================================
# DETERMINISTIC SIMULATION TESTS
# =========================================================================
if __name__ == "__main__":
    print("==================================================================")
    print(" SpaceShield Closed-Loop: Dynamic Overlap Window Manager Validation")
    print("==================================================================")
    
    manager = DynamicOverlapWindowManager()
    
    # Initialize mock active buffer of size 64 filled with consecutive values to trace positions
    # (e.g. index representation)
    mock_buf = np.zeros((64, 4), dtype=np.complex64)
    for t in range(64):
        mock_buf[t, :] = float(t) + 1j * float(t)
        
    # 1. Nominal 32 -> 64 Transition
    print("[*] Scenario 1: Transition FFT 32 -> 64...")
    # The active buffer was size 32. We want to transition to size 64.
    # The last D = 32 samples of history need to be retained at the top of the new buffer.
    # From index (32 - 32) = 0 to 31.
    manager.active_fft_size = 32
    manager.active_discard = 16
    
    new_D = manager.transition_block_size(mock_buf, 64)
    print(f"    -> New Discard Length D: {new_D} | Top sample in buffer: {mock_buf[0, 0]}")
    assert new_D == 32, "Incorrect discard length mapped for FFT 64!"
    assert mock_buf[0, 0] == 0.0 + 0.0j, "Failed to copy old history boundary correctly!"
    print("    -> 32 to 64 transition: [PASSED]")
    
    # 2. Transition 64 -> 16
    print("\n[*] Scenario 2: Transition FFT 64 -> 16...")
    # Reset mock_buf with values
    for t in range(64):
        mock_buf[t, :] = float(t) + 1j * float(t)
    manager.active_fft_size = 64
    manager.active_discard = 32
    
    new_D = manager.transition_block_size(mock_buf, 16)
    # The new discard is D = 8.
    # We copy the last 8 samples from the old buffer (indices 56 to 63) to the top (0 to 7) of the new buffer.
    print(f"    -> New Discard Length D: {new_D} | Top sample: {mock_buf[0, 0]} | Sample 7: {mock_buf[7, 0]}")
    assert new_D == 8, "Incorrect discard length mapped for FFT 16!"
    assert mock_buf[0, 0] == 56.0 + 56.0j, "Failed to align oldest boundary history during downscale!"
    print("    -> 64 to 16 transition: [PASSED]")
    
    # 3. Transition 16 -> 32
    print("\n[*] Scenario 3: Transition FFT 16 -> 32...")
    for t in range(64):
        mock_buf[t, :] = float(t) + 1j * float(t)
    manager.active_fft_size = 16
    manager.active_discard = 8
    
    new_D = manager.transition_block_size(mock_buf, 32)
    # The new discard is D = 16.
    # We copy the last 16 samples from the old buffer (indices 0 to 15, since old size was 16)
    print(f"    -> New Discard Length D: {new_D} | Top sample: {mock_buf[0, 0]}")
    assert new_D == 16, "Incorrect discard length mapped for FFT 32!"
    assert mock_buf[0, 0] == 0.0 + 0.0j, "Failed to align oldest boundary history during upscale!"
    print("    -> 16 to 32 transition: [PASSED]")

    print("\n[+] Dynamic overlap window manager validation complete.")
