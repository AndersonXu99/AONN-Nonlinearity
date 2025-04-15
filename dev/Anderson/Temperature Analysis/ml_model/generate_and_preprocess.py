import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Append ElecSus path
sys.path.append(os.path.abspath("ElecSus-main"))

from elecsus.libs.spectra import get_spectra

# === Configuration ===
T_CELSIUS = 70
DETUNING_GHZ = np.linspace(-10, 10, 2000) * 1e3  # GHz, as in your SAS spectrum
E_in = np.array([1, 0, 0])              # Horizontal linear polarization

p_dict = {
    'Elem': 'Rb',
    'Dline': 'D1',       # Match 795 nm line
    'T': T_CELSIUS,
    'lcell': 70e-3,
    'Bfield': 0,
    'Btheta': 0,
    'rb85frac': 0
}

# === Run ElecSus
S0, *_ = get_spectra(DETUNING_GHZ, E_in, p_dict, outputs=['S0'])

# === Plot
plt.figure(figsize=(10, 5))
plt.plot(DETUNING_GHZ, S0, color='red')
plt.title(f"Simulated Rb D1 Spectrum at T = {T_CELSIUS}°C (Pure 87Rb)")
plt.xlabel("Detuning (GHz)")
plt.ylabel("Transmission (S0)")
plt.grid(True)
plt.tight_layout()
plt.show()
