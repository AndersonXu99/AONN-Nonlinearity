import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# === ElecSus Setup ===
import sys
sys.path.append(os.path.abspath("ElecSus-main"))
from elecsus.libs.spectra import get_spectra

# === Linear calibration function ===
def linear_map(V, a, b):
    return a * V + b

# === Main processing function ===
def process_rb_spectrum(doppler_file, baseline_file, sim_temperature=70):
    # Load CSVs
    doppler_df = pd.read_csv(doppler_file, delimiter=';')
    baseline_df = pd.read_csv(baseline_file, delimiter=';')

    # Clean column names
    doppler_df.columns = [col.strip().replace('"', '') for col in doppler_df.columns]
    baseline_df.columns = [col.strip().replace('"', '') for col in baseline_df.columns]
    doppler_df = doppler_df.drop(columns=["Unnamed: 3"], errors="ignore")
    baseline_df = baseline_df.drop(columns=["Unnamed: 3"], errors="ignore")

    # # === Plot raw signals before normalization ===
    # plt.figure(figsize=(10, 5))
    # plt.plot(doppler_df["Piezo Voltage (V)"], doppler_df["Fine In 1 (V)"], label="Fine In 1 (SAS)", lw=1.5)
    # plt.plot(doppler_df["Piezo Voltage (V)"], doppler_df["Fine In 2 (V)"], label="Fine In 2 (Doppler)", lw=1.5)
    # plt.xlabel("Piezo Voltage (V)")
    # plt.ylabel("Signal Voltage (V)")
    # plt.title("Raw Experimental Spectra Before Normalization")
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # === Normalize Doppler-broadened signal using baseline
    baseline_interp = interp1d(
        baseline_df["Piezo Voltage (V)"],
        baseline_df["Fine In 2 (V)"],
        kind='linear',
        fill_value='extrapolate'
    )
    interp_baseline = baseline_interp(doppler_df["Piezo Voltage (V)"].values)
    normalized_transmission = doppler_df["Fine In 2 (V)"].values / interp_baseline

    # === Use known SAS peaks for calibration
    V1 = 43.631035
    V2 = 57.066822
    D1 = -6834.682 / 2  # MHz
    D2 = +6834.682 / 2

    selected_voltages = np.array([V1, V2])
    known_detunings = np.array([D1, D2])
    fit_params, _ = curve_fit(linear_map, selected_voltages, known_detunings)
    a_fit, b_fit = fit_params

    detuning_axis = linear_map(doppler_df["Piezo Voltage (V)"].values, a_fit, b_fit)

    print(f"Using fixed calibration: a = {a_fit:.6f} MHz/V, b = {b_fit:.2f} MHz")
    print(f"    SAS peaks → Detuning: {linear_map(V1, a_fit, b_fit):.2f} MHz, {linear_map(V2, a_fit, b_fit):.2f} MHz")

    # === Final processed dataframe
    processed = pd.DataFrame({
        "Detuning (MHz)": detuning_axis,
        "Normalized Transmission": normalized_transmission
    }).sort_values("Detuning (MHz)").reset_index(drop=True)

    # === Generate ElecSus spectrum
    detuning_sim = np.linspace(-10e3, 10e3, 2000)  # MHz
    E_in = np.array([1, 0, 0])
    p_dict = {
        'Elem': 'Rb',
        'Dline': 'D1',
        'T': sim_temperature,
        'lcell': 70e-3,
        'GammaBuf': 800,
        'shift': -800,
        'Bfield': 0,
        'Btheta': 0,
        'rb85frac': 0
    }
    S0, *_ = get_spectra(detuning_sim, E_in, p_dict, outputs=['S0'])

    # === Plot experimental + simulated overlay
    plt.figure(figsize=(10, 5))
    plt.plot(processed["Detuning (MHz)"] / 1e3, processed["Normalized Transmission"], label="Experimental (Normalized)", lw=2)
    plt.plot(detuning_sim / 1e3, S0, label=f"Simulated ElecSus ({sim_temperature}°C)", lw=2, linestyle='--')
    plt.title("Normalized Transmission Spectrum (Rb-87 D1) with ElecSus Overlay")
    plt.xlabel("Detuning (GHz)")
    plt.ylabel("Transmission")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return processed

# === Input paths ===
doppler_file_path = "Experimental Data/April 14 2025/122 mW 77.8C_TC 87C_Thermometer.csv"
baseline_file_path = "Experimental Data/April 14 2025/122 mW Baseline.csv"

# === Run the pipeline ===
if __name__ == "__main__":
    result_df = process_rb_spectrum(doppler_file_path, baseline_file_path, sim_temperature=50)
    result_df.to_csv("Experimental Data/April 14 2025/Processed_122mW_Exp.csv", index=False)
    print("\n✅ Processed experimental spectrum saved.")
