#!/usr/bin/env python3
# redpitaya_acquirer.py

import time
import numpy as np
from redpitaya_scpi import scpi

class RedPitayaAcquirer:
    """
    Encapsulates connection, configuration, and data acquisition
    from a Red Pitaya via its SCPI interface.
    """

    def __init__(self, ip: str, port: int = 5000, decimation: int = 1024, avg_off: bool = True):
        """
        Establish SCPI connection and configure acquisition parameters.

        Args:
            ip:         Red Pitaya IP address or mDNS name.
            port:       SCPI port (default 5000).
            decimation: Decimation factor (125 MS/s ÷ decimation ≈ sample rate).
            avg_off:    If True, disable on-FPGA averaging.
        Raises:
            ConnectionError: if unable to connect.
        """
        try:
            self.rp = scpi(ip, port=port)
            print(f"Connected to Red Pitaya at {ip}:{port}")
        except Exception as e:
            raise ConnectionError(f"Could not connect to {ip}:{port} → {e}")

        # configure decimation
        self.rp.tx_txt(f'ACQ:DEC {decimation}')
        if avg_off:
            self.rp.tx_txt('ACQ:AVG OFF')

    def _wait_for_trigger(self):
        """Arm acquisition and wait until the buffer is ready ('TD')."""
        self.rp.tx_txt('ACQ:START')
        self.rp.tx_txt('ACQ:TRIG NOW')
        while True:
            self.rp.tx_txt('ACQ:TRIG:STAT?')
            if self.rp.rx_txt().strip() == 'TD':
                return

    def acquire(self, duration_s: float, samples_per_read: int = 1000):
        """
        Acquire contiguous samples from both channels for a given duration,
        then return the mean voltage on each channel.

        Args:
            duration_s:        total acquisition time in seconds
            samples_per_read:  how many samples to fetch per loop

        Returns:
            (mean_ch1, mean_ch2): tuple of floats
        """
        # arm & trigger
        self._wait_for_trigger()

        data_ch1, data_ch2 = [], []
        t_start = time.time()

        while (time.time() - t_start) < duration_s:
            block1 = self.rp.acq_data(1, num_samples=samples_per_read,
                                     lat=True, convert=True)
            block2 = self.rp.acq_data(2, num_samples=samples_per_read,
                                     lat=True, convert=True)
            data_ch1.extend(block1)
            data_ch2.extend(block2)

        mean_ch1 = np.mean(data_ch1)
        mean_ch2 = np.mean(data_ch2)
        return mean_ch1, mean_ch2
