import numpy as np
import matplotlib.pyplot as plt

# Constants
epsilon_0 = 8.854e-12  # Vacuum permittivity (F/m)
hbar = 1.054e-34       # Reduced Planck's constant (J·s)
mu_13 = mu_23 = 2.5377e-29  # Dipole moment (C·m)

Gamma3 = 2 * np.pi * 6e6   # Decay rate from level 3 (Hz)
Gamma31 = 5 / 9 * Gamma3   # Decay from level 3 to level 1 (Hz)
Gamma32 = Gamma23 = 4 / 9 * Gamma3   # Decay from level 3 to level 2 (Hz)
Gamma12 = Gamma21 = 0.05 * Gamma3    # Decay rates between levels 1 and 2 (Hz)
gamma13 = (Gamma3 + Gamma12) / 2  # Decoherence rate between levels 1 and 3 (Hz)
gamma23 = (Gamma3 + Gamma21) / 2  # Decoherence rate between levels 2 and 3 (Hz)
gamma12 = (Gamma12 + Gamma21) / 2  # Decoherence rate between ground states (Hz)

w_1 = 2 * np.pi * 3e8 / 795e-9  # Angular frequency (rad/s)
k_0 = w_1 / 3e8  # Wave vector (m^-1)
L = 0.1          # Medium length (m)
z = 0.0005       # Step size in meters


# Function to calculate optical depth and EIT behavior
def calculate_OD_1(Om_1=1.0, delta_1=0.0, delta_2=0.0, OD_0=10):

    factor = ((4 * np.pi * mu_13**2 * L) / (Gamma3 * hbar * epsilon_0 * 795e-9))
    N = OD_0 / factor

    print (f"Optical Depth: {N}")

    n_rabi = 500
    Om_2_list_adapted = np.linspace(0.005, 12, n_rabi, dtype=complex)  # Coupling Rabi frequency range (dimensionless)
    Om_2_list = Om_2_list_adapted * Gamma3
    Om_1_list = np.full(n_rabi, Om_1 * Gamma3, dtype=complex)

    nz = int(L / z)  # Number of z steps
    z_array = np.arange(nz) * z  # Array of z values

    # Initialize arrays to store populations
    rho11_values = np.zeros((nz, n_rabi), dtype=float)
    rho22_values = np.zeros((nz, n_rabi), dtype=float)
    rho33_values = np.zeros((nz, n_rabi), dtype=float)
    rho31_values = np.zeros((nz, n_rabi), dtype=complex)
    rho32_values = np.zeros((nz, n_rabi), dtype=complex)

    for i in range(nz):
        for j in range(len(Om_2_list)):
            Om_2_actual = Om_2_list[j]
            Om_1_actual = Om_1_list[j]
            delta_1_actual = delta_1 * Gamma3
            delta_2_actual = delta_2 * Gamma3

            I = 1j
            # Construct the matrix A
            A = np.array([
                # Row 1
                [Gamma31 + Gamma12, 0, I*Om_1_actual/2, 0, Gamma31 - Gamma21, 0, -I*np.conjugate(Om_1_actual)/2, 0],
                # Row 2
                [0, gamma12 - I*(delta_2_actual - delta_1_actual), I*Om_2_actual/2, 0, 0, 0, 0, -I*np.conjugate(Om_1_actual)/2],
                # Row 3
                [I*np.conjugate(Om_1_actual), I*np.conjugate(Om_2_actual)/2, gamma13 + I*delta_1_actual, 0, I*np.conjugate(Om_1_actual)/2, 0, 0, 0],
                # Row 4
                [0, 0, 0, gamma12 + I*(delta_2_actual - delta_1_actual), 0, I*Om_1_actual/2, -I*np.conjugate(Om_2_actual)/2, 0],
                # Row 5
                [Gamma32 - Gamma12, 0, 0, 0, Gamma32 + Gamma21, I*Om_2_actual/2, 0, -I*np.conjugate(Om_2_actual)/2],
                # Row 6
                [I*np.conjugate(Om_2_actual)/2, 0, 0, I*np.conjugate(Om_1_actual)/2, I*np.conjugate(Om_2_actual), gamma23 + I*delta_2_actual, 0, 0],
                # Row 7
                [-I*Om_1_actual, 0, 0, -I*Om_2_actual/2, -I*Om_1_actual/2, 0, gamma13 - I*delta_1_actual, 0],
                # Row 8
                [-I*Om_2_actual/2, -I*Om_1_actual/2, 0, 0, -I*Om_2_actual, 0, 0, gamma23 - I*delta_2_actual]
            ], dtype=complex)
            
            # Construct the vector b
            b = np.array([
                Gamma31,
                0,
                I*np.conjugate(Om_1_actual)/2,
                0,
                Gamma32,
                I*np.conjugate(Om_2_actual)/2,
                -I*Om_1_actual/2,
                -I*Om_2_actual/2
            ], dtype=complex)

            try:
                v = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                rho11_values[i][j] = np.nan
                rho22_values[i][j] = np.nan
                rho33_values[i][j] = np.nan
                rho31_values[i][j] = np.nan
                rho32_values[i][j] = np.nan
                continue

            # Extract the solutions
            rho11 = np.real(v[0])
            rho12 = v[1]
            rho13 = v[2]
            rho21 = v[3]
            rho22 = np.real(v[4])
            rho23 = v[5]
            rho31 = v[6]
            rho32 = v[7]

            # Compute rho33
            rho33 = 1 - rho11 - rho22

            # Store the populations
            rho11_values[i][j] = np.real(rho11)
            rho22_values[i][j] = np.real(rho22)
            rho33_values[i][j] = np.real(rho33)
            rho31_values[i][j] = rho31
            rho32_values[i][j] = rho32
            

            # Update the Rabi frequencies
            Om_2_list[j] = Om_2_actual + z * (1j) * (k_0 / (epsilon_0 * hbar)) * N * rho32 * mu_23**2 
            Om_1_list[j] = Om_1_actual + z * (1j) * (k_0 / (epsilon_0 * hbar)) * N * rho31 * mu_13**2 

    return Om_2_list_adapted, Om_2_list, Om_1_list


def plot_results_with_OD_sweep(Om_1, delta_1, delta_2, OD_values):
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    for OD_0 in OD_values:
        Om_2_list_adapted, Om_2_list, Om_1_list = calculate_OD_1(Om_1, delta_1, delta_2, OD_0)

        Transparency_Omega_1 = np.abs(Om_1_list)**2 / np.abs(Om_1_list[0])**2
        Transparency_Omega_2 = np.abs(Om_2_list)**2 / np.abs(Om_2_list[0])**2

        Output_Intensity_Omega_1 = Transparency_Omega_1 * np.abs(Om_1_list[0])**2 / (Gamma3**2)
        Output_Intensity_Omega_2 = Transparency_Omega_2 * np.abs(Om_2_list[0])**2 / (Gamma3**2)

        label = f"OD = {OD_0}"

        # Plot Output Intensity of Omega_1 vs Initial Omega_2
        ax[0].plot(np.real(Om_2_list_adapted)**2, Output_Intensity_Omega_1, label=label)
        ax[1].plot(np.real(Om_2_list_adapted)**2, Output_Intensity_Omega_2, label=label)

    ax[0].set_xlabel('Initial Intensity 2 ($|\\Omega_{2,in} / \\Gamma_3|^2$)')
    ax[0].set_ylabel('Output Intensity 1 ($|\\Omega_{1,out} / \\Gamma_3|^2$)')
    ax[0].set_title('Output Intensity 1 vs Initial Intensity 2')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].set_xlabel('Initial Intensity 2 ($|\\Omega_{2,in} / \\Gamma_3|^2$)')
    ax[1].set_ylabel('Output Intensity 2 ($|\\Omega_{2,out} / \\Gamma_3|^2$)')
    ax[1].set_title('Output Intensity 2 vs Initial Intensity 2')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

# Call the function with desired parameters
if __name__ == "__main__":
    Om_1 = 3.0
    delta_1 = 0.5
    delta_2 = 0.0
    OD_values = [1, 10, 100, 1000]  # Sweep through OD values
    plot_results_with_OD_sweep(Om_1, delta_1, delta_2, OD_values)






