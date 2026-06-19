import time
import threading
import logging
import numpy as np
import SoapySDR
from SoapySDR import * # type: ignore
import queue
import scipy.signal
from rf_frontend_emulator import RfFrontendEmulator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [SoapyBridge] %(message)s')
logger = logging.getLogger(__name__)

class SoapyReceiverBridge:
    """
    A robust, zero-allocation hardware ingestion bridge using SoapySDR.
    Intercepts physical antenna streams and pipes them directly into the Spatial Hardware Harness queue.
    """
    def __init__(self, 
                 target_queue: queue.Queue,
                 energy_orchestrator = None,
                 freq_mhz: float = 1176.45, # Default to NavIC L5
                 sample_rate_msps: float = 2.0,
                 gain_db: float = 40.0,
                 num_channels: int = 4,
                 chunk_size: int = 8192,
                 device_args: dict = {"driver": "rtlsdr"},
                 rf_emulator: RfFrontendEmulator = None):
        
        self.target_queue = target_queue
        self.energy_orchestrator = energy_orchestrator
        self.freq = freq_mhz * 1e6
        self.sample_rate = sample_rate_msps * 1e6
        self.gain = gain_db
        self.num_channels = num_channels
        self.chunk_size = chunk_size
        self.device_args = device_args
        self.rf_emulator = rf_emulator
        
        self.sdr = None
        self.rx_stream = None
        self.is_running = False
        self._thread = None
        
        # Pre-allocate zero-copy buffers for physical ingestion
        # We allocate a flat array for SoapySDR API, which expects a list of pointers.
        self._buffers = [np.empty(self.chunk_size, dtype=np.complex64) for _ in range(self.num_channels)]
        self._buffer_pointers = [buf for buf in self._buffers]
        
        # Power Management Tracking
        self._current_mode = "NOMINAL_RENEWABLE_MODE"
        
        # Pre-allocate static phase-smoothing transition window (Hann curve)
        self._transition_window = np.hanning(self.chunk_size).astype(np.float32)
        
    def initialize_hardware(self):
        """Bind to the SDR hardware and configure stream parameters."""
        logger.info(f"Initializing SDR device with args: {self.device_args}")
        try:
            self.sdr = SoapySDR.Device(self.device_args)
        except Exception as e:
            logger.error(f"Failed to initialize SDR device: {e}")
            raise
            
        # Configure channels
        for ch in range(self.num_channels):
            try:
                self.sdr.setSampleRate(SOAPY_SDR_RX, ch, self.sample_rate)
                self.sdr.setFrequency(SOAPY_SDR_RX, ch, self.freq)
                self.sdr.setGain(SOAPY_SDR_RX, ch, self.gain)
                
                # Setup antenna if necessary (driver specific)
                antennas = self.sdr.listAntennas(SOAPY_SDR_RX, ch)
                if antennas:
                    self.sdr.setAntenna(SOAPY_SDR_RX, ch, antennas[0])
            except Exception as e:
                logger.warning(f"Could not fully configure channel {ch}. It may not exist on this device. Error: {e}")
                
        logger.info(f"Hardware configured: Freq={self.freq/1e6}MHz, FS={self.sample_rate/1e6}MSPS, Gain={self.gain}dB")
        
        # Setup the reception stream
        channels = list(range(self.num_channels))
        try:
            self.rx_stream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, channels)
        except Exception as e:
            logger.error(f"Failed to setup multi-channel stream: {e}")
            raise

    def start(self):
        """Start the threaded hardware capture loop."""
        if self.is_running:
            return
            
        if not self.sdr or not self.rx_stream:
            self.initialize_hardware()
            
        self.is_running = True
        self.sdr.activateStream(self.rx_stream)
        logger.info("SDR Stream activated. Engaging fast-path ingestion loop.")
        
        self._thread = threading.Thread(target=self._capture_loop, name="SoapyIngestLoop", daemon=True)
        self._thread.start()

    def stop(self):
        """Safely wind down the hardware stream and thread."""
        logger.info("Stopping SoapySDR capture loop...")
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            
        if self.sdr and self.rx_stream:
            self.sdr.deactivateStream(self.rx_stream)
            self.sdr.closeStream(self.rx_stream)
            logger.info("SDR Stream safely deactivated.")

    def _capture_loop(self):
        """High-priority, zero-allocation ring-buffer execution block."""
        
        # Flags
        OVERFLOW_FLAG = SoapySDR.SOAPY_SDR_OVERFLOW
        TIMEOUT_FLAG = SoapySDR.SOAPY_SDR_TIMEOUT
        
        last_time = time.time()
        
        while self.is_running:
            # 1. TBPM Flag Evaluation
            decimation_factor = 1
            active_sample_rate = self.sample_rate
            
            if self.energy_orchestrator:
                tbpm_cfg = self.energy_orchestrator.get_atomic_config()
                target_mode = tbpm_cfg.get("power_state", "NOMINAL_RENEWABLE_MODE")
                
                if target_mode == "CRITICAL_SUSTAINABILITY_MODE":
                    # Decimate to 1/4 rate (e.g. 4MHz -> 1MHz)
                    decimation_factor = int(self.sample_rate / tbpm_cfg.get("sdr_ingest_rate_hz", self.sample_rate))
                    if decimation_factor < 1: decimation_factor = 1
                    active_sample_rate = self.sample_rate / decimation_factor
                    
                    # Dynamically scale CPU loop sleep threshold 
                    # If we skip processing frames natively in hardware, we sleep to free CPU scheduler
                    time.sleep(0.001 * decimation_factor)
                    
            # Detect operational state shift to trigger phase-smoothing interpolation
            mode_shifted = False
            if self.energy_orchestrator and self._current_mode != target_mode:
                mode_shifted = True
                self._current_mode = target_mode
                logger.warning(f"[SDR Bridge] TBPM Transition intercepted! Engaging {self._current_mode} logic.")

            # Block until data is available, with a very tight timeout
            sr = self.sdr.readStream(self.rx_stream, self._buffer_pointers, self.chunk_size, timeoutUs=100000)
            
            # Trap hardware errors and metadata
            if sr.ret < 0:
                if sr.ret == TIMEOUT_FLAG:
                    continue # Standard timeout, just poll again
                elif sr.ret == OVERFLOW_FLAG:
                    logger.warning("[!] HARDWARE OVERFLOW DETECTED: DSP Pipeline falling behind ingestion.")
                else:
                    logger.error(f"[!] STREAM ERROR CODE: {sr.ret}")
                continue
            
            if sr.ret != self.chunk_size:
                logger.warning(f"Sequence Discontinuity: Received partial block ({sr.ret}/{self.chunk_size})")
                continue
            
            # Fast-Path: Stack independent channel arrays coherently
            coherent_block = np.vstack(self._buffers)
            
            # Intercept and inject RF frontend impairments in-place if emulator is active
            if self.rf_emulator is not None:
                self.rf_emulator.emulate_impairments(coherent_block)
            
            # 2. Transition Digital Interpolation Window
            # If the mode just shifted, apply a continuous Hann smoothing envelope to the block
            # to gracefully slope the phase transitions and prevent the Carrier Lock PLL from snapping.
            if mode_shifted:
                coherent_block *= self._transition_window
                
            # 3. Vector Decimation Engine (Zero-Allocation Memory View Extraction)
            if decimation_factor > 1:
                # Extract every N-th complex64 sample purely via NumPy memory striding
                processed_block = coherent_block[:, ::decimation_factor]
            else:
                processed_block = coherent_block
            
            payload = {
                "iq_data": processed_block,
                "timestamp": time.time(),
                "center_freq": self.freq,
                "sample_rate": active_sample_rate
            }
            
            # Push to the synchronized pipeline queue
            try:
                self.target_queue.put_nowait(payload)
            except queue.Full:
                logger.warning("Pipeline ingestion queue full. Dropping physical block.")
                
            # Optional: Telemetry tracking
            now = time.time()
            if now - last_time > 5.0:
                logger.info(f"Heartbeat: Stream active at {self.sample_rate/1e6} MSPS. Queue depth: {self.target_queue.qsize()}")
                last_time = now

if __name__ == "__main__":
    # Rapid Verification & Execution Stub
    print("[*] Testing SoapySDR Receiver Bridge Compilation...")
    test_queue = queue.Queue(maxsize=100)
    
    # Instantiate emulator and bridge for testing
    emulator = RfFrontendEmulator(sample_rate_hz=2.0e6)
    bridge = SoapyReceiverBridge(
        target_queue=test_queue, 
        chunk_size=8192,
        rf_emulator=emulator
    )
    try:
        bridge.initialize_hardware()
        bridge.start()
        time.sleep(2)
        bridge.stop()
    except Exception as e:
        print(f"[-] Hardware verification bypass (No SDR attached?): {e}")
