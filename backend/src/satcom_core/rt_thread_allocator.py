"""
Task 41.1: Bare-Metal Hardware Orchestration Utility
Low-Level Embedded Linux Kernel Thread & Memory Pinning
"""

import sys
import os
import ctypes
import multiprocessing

# ============================================================================
# POSIX / Linux System Call Constants
# ============================================================================
MCL_CURRENT = 1
MCL_FUTURE  = 2

SCHED_FIFO  = 1
SCHED_MAX_PRIORITY = 99

CPU_SETSIZE = 1024
NCPUBITS = 8 * ctypes.sizeof(ctypes.c_ulong)

# ============================================================================
# Low-Level C-Types Structures
# ============================================================================
class SchedParam(ctypes.Structure):
    """POSIX sched_param structure for sched_setscheduler"""
    _fields_ = [("sched_priority", ctypes.c_int)]

class CpuSet(ctypes.Structure):
    """POSIX cpu_set_t bitmask for sched_setaffinity"""
    _fields_ = [("__bits", ctypes.c_ulong * (CPU_SETSIZE // NCPUBITS))]
    
    def zero(self):
        """Zero out the CPU bitmask (CPU_ZERO)"""
        for i in range(CPU_SETSIZE // NCPUBITS):
            self.__bits[i] = 0
            
    def set(self, cpu):
        """Set a specific CPU bit (CPU_SET)"""
        if 0 <= cpu < CPU_SETSIZE:
            self.__bits[cpu // NCPUBITS] |= (1 << (cpu % NCPUBITS))


class RTThreadAllocator:
    """
    Zero-allocation Real-Time Orchestrator.
    Must be executed immediately upon system boot before memory buffers are mapped.
    """
    def __init__(self):
        self.is_linux = sys.platform.startswith('linux')
        self.libc = None
        
        if self.is_linux:
            try:
                self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
                
                # Bind function signatures to prevent argument conversion allocations during runtime
                self.libc.mlockall.argtypes = [ctypes.c_int]
                self.libc.mlockall.restype = ctypes.c_int
                
                self.libc.sched_setscheduler.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(SchedParam)]
                self.libc.sched_setscheduler.restype = ctypes.c_int
                
                self.libc.sched_setaffinity.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.POINTER(CpuSet)]
                self.libc.sched_setaffinity.restype = ctypes.c_int
            except OSError as e:
                print(f"[WARN] Failed to load libc.so.6: {e}. RT Orchestration disabled.")
                self.libc = None

    def discover_topology(self):
        """
        Programmatically discovers the host core topology.
        Returns a list of isolated cores for DSP stride workers (excluding core 0).
        """
        num_cores = multiprocessing.cpu_count()
        if num_cores > 1:
            # Isolate all available physical cores excluding Core 0 (reserved for OS/Network)
            return list(range(1, num_cores))
        return [0]

    def enforce_memory_lock(self):
        """
        Executes mlockall() to permanently pin the virtual memory address space 
        into physical RAM, completely preventing page faults.
        """
        if not self.libc:
            return False
            
        result = self.libc.mlockall(MCL_CURRENT | MCL_FUTURE)
        if result != 0:
            errno = ctypes.get_errno()
            print(f"[ERROR] mlockall(MCL_CURRENT|MCL_FUTURE) failed. Errno: {errno}. Requires CAP_IPC_LOCK.")
            return False
            
        print("[INFO] Virtual Address Space successfully pinned to Physical RAM (mlockall).")
        return True

    def enforce_rt_scheduler(self):
        """
        Promotes the process to the SCHED_FIFO real-time scheduler class
        with maximum POSIX execution priority (99).
        """
        if not self.libc:
            return False
            
        param = SchedParam(sched_priority=SCHED_MAX_PRIORITY)
        result = self.libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(param))
        
        if result != 0:
            errno = ctypes.get_errno()
            print(f"[ERROR] sched_setscheduler(SCHED_FIFO, 99) failed. Errno: {errno}. Requires CAP_SYS_NICE.")
            return False
            
        print(f"[INFO] Process elevated to SCHED_FIFO Real-Time class (Priority {SCHED_MAX_PRIORITY}).")
        return True

    def pin_thread_affinity(self, core_list):
        """
        Pins the current execution context strictly to the provided physical cores
        using sched_setaffinity.
        """
        if not self.libc:
            return False
            
        if not core_list:
            return False
            
        mask = CpuSet()
        mask.zero()
        for core_id in core_list:
            mask.set(core_id)
            
        result = self.libc.sched_setaffinity(0, ctypes.sizeof(mask), ctypes.byref(mask))
        if result != 0:
            errno = ctypes.get_errno()
            print(f"[ERROR] sched_setaffinity failed. Errno: {errno}.")
            return False
            
        print(f"[INFO] Execution Affinity successfully pinned to Physical Cores: {core_list}")
        return True

    def orchestrate_bare_metal(self):
        """
        Master routine to lock memory, elevate scheduler, and pin physical cores.
        Handles graceful fallback on non-Linux platforms (e.g. Windows dev environments).
        """
        print("===================================================================")
        print("SPACESHIELD BARE-METAL ORCHESTRATOR INITIALIZING...")
        print("===================================================================")
        
        if not self.is_linux:
            print("[WARN] Non-Linux Kernel detected. RT Thread Allocator operating in passthrough mode.")
            return
            
        cores = self.discover_topology()
        
        # 1. Lock Memory (mlockall)
        self.enforce_memory_lock()
        
        # 2. Pin CPU Affinity (sched_setaffinity)
        self.pin_thread_affinity(cores)
        
        # 3. Elevate to Real-Time Class (sched_setscheduler)
        self.enforce_rt_scheduler()
        
        print("===================================================================")
        print("BARE-METAL ORCHESTRATION COMPLETE.")
        print("===================================================================")


# Static singleton execution upon module load if required
_orchestrator = RTThreadAllocator()

def boot_orchestrator():
    """Entry point for system boot sequence."""
    _orchestrator.orchestrate_bare_metal()

if __name__ == "__main__":
    # If run as a standalone script, attempt execution immediately.
    boot_orchestrator()
