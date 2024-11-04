import numpy as np
import matplotlib.pyplot as plt

# Define the system of differential equations
def system(t, y, params):
    # Extract variables from y (each rho is complex)
    rho11, rho12, rho13, rho22, rho23, rho33 = y

    # rho21 = conjugate of rho12, rho31 = conjugate of rho13, rho32 = conjugate of rho23
    rho21 = np.conj(rho12)
    rho31 = np.conj(rho13)
    rho32 = np.conj(rho23)

    # Extract parameters
    Om_p, Om_c, Gamma12, Gamma21, Gamma31, Gamma32, gamma13, gamma23, gamma12, delta_p, delta_c = params

    Gamma3_total = Gamma31 + Gamma32  # Total decay rate of the excited state

    # Population equations
    drho11_dt = Gamma31 * rho33 + Gamma21 * rho22 - Gamma12 * rho11 - (1j / 2) * (Om_p * rho13 - np.conj(Om_p) * rho31)

    drho22_dt = Gamma32 * rho33 + Gamma12 * rho11 - Gamma21 * rho22 - (1j / 2) * (Om_c * rho23 - np.conj(Om_c) * rho32)

    drho33_dt = -Gamma3_total * rho33 + (1j / 2) * (Om_p * rho13 + Om_c * rho23 - np.conj(Om_p) * rho31 - np.conj(Om_c) * rho32)

    # Coherence equations
    drho12_dt = -gamma12 * rho12 + 1j * (delta_p - delta_c) * rho12 + (1j / 2) * (np.conj(Om_p) * rho32 - Om_c * rho13)

    drho13_dt = -gamma13 * rho13 - 1j * delta_p * rho13 + (1j / 2) * (np.conj(Om_p) * (rho33 - rho11) - np.conj(Om_c) * rho12)

    drho23_dt = -gamma23 * rho23 - 1j * delta_c * rho23 + (1j / 2) * (np.conj(Om_c) * (rho33 - rho22) - np.conj(Om_p) * rho21)

    # Return the derivatives of the state variables
    return np.array([drho11_dt, drho12_dt, drho13_dt, drho22_dt, drho23_dt, drho33_dt])

# RK4 method for integrating the system of ODEs with population correction
def rk4_step(f, t, y, dt, params):
    k1 = f(t, y, params)
    k2 = f(t + dt / 2, y + dt / 2 * k1, params)
    k3 = f(t + dt / 2, y + dt / 2 * k2, params)
    k4 = f(t + dt, y + dt * k3, params)

    # Calculate the next step
    y_next = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Enforce non-negative populations for rho11, rho22, and rho33
    y_next[0] = max(0, np.real(y_next[0]))  # Ensure rho11 is non-negative
    y_next[3] = max(0, np.real(y_next[3]))  # Ensure rho22 is non-negative
    y_next[5] = max(0, np.real(y_next[5]))  # Ensure rho33 is non-negative

    # Normalize the populations to maintain total population = 1
    total_population = y_next[0] + y_next[3] + y_next[5]
    if total_population > 0:  # Prevent division by zero
        y_next[0] /= total_population
        y_next[3] /= total_population
        y_next[5] /= total_population

    return y_next

# Initialize parameters (units: MHz for frequencies, μs for time)
Om_p = 100       # Probe laser Rabi frequency (MHz). Frequency = 377109307.2 MHz
Om_c = 175       # Coupling laser Rabi frequency (MHz). Frequency = 377108193 MHz
delta_p = 0    # Detuning for probe laser (MHz)
delta_c = 0    # Detuning for coupling laser (MHz)
Gamma3 = 2 * np.pi * 6   # (MHz)
Gamma31 = 5/9 * Gamma3   # Decay from level 3 to level 1 (MHz)
Gamma32 = 4/9 * Gamma3   # Decay from level 3 to level 2 (MHz)
Gamma12 = 0.00000001 * Gamma3  # Decay rate between levels 1 and 2 (MHz)
Gamma21 = 0.00000001* Gamma3  # Decay rate between levels 2 and 1 (MHz)
gamma13 = (Gamma3 + Gamma12)/2    # Decoherence rate between levels 1 and 3 (MHz)
gamma23 = (Gamma3 + Gamma21)/2    # Decoherence rate between levels 2 and 3 (MHz)
gamma12 = (Gamma12 + Gamma21)/2  # Decoherence rate between ground states (MHz)

params = [Om_p, Om_c, Gamma12, Gamma21, Gamma31, Gamma32, gamma13, gamma23, gamma12, delta_p, delta_c]

# Initial conditions (assumed populations for rho11, rho22, and rho33)
rho11_0 = 0.5  # Initial population in state 1
rho12_0 = 0.0 + 0j  # Coherence between states 1 and 2
rho13_0 = 0.0 + 0j  # Coherence between states 1 and 3
rho22_0 = 0.5  # Initial population in state 2
rho23_0 = 0.0 + 0j  # Coherence between states 2 and 3
rho33_0 = 0.0  # Initial population in state 3
y0 = np.array([rho11_0, rho12_0, rho13_0, rho22_0, rho23_0, rho33_0])

# Time settings
t0 = 0
tf = 0.5    # Total time in microseconds
dt = 0.0001  # Time step in microseconds
time_points = np.arange(t0, tf, dt)

# Storage for solutions
solutions = np.zeros((len(time_points), len(y0)), dtype=complex)
solutions[0, :] = y0

# Time integration using RK4 with population correction
for i in range(1, len(time_points)):
    t = time_points[i - 1]
    y = solutions[i - 1, :]
    solutions[i, :] = rk4_step(system, t, y, dt, params)

# Plot the results

# Populations
plt.figure(figsize=(12, 10))

# First subplot: Populations
plt.subplot(4, 1, 1)
plt.plot(time_points, np.real(solutions[:, 0]), label=r'$\rho_{11}$')
plt.plot(time_points, np.real(solutions[:, 3]), label=r'$\rho_{22}$')
plt.plot(time_points, np.real(solutions[:, 5]), label=r'$\rho_{33}$')
plt.xlabel('Time (μs)')
plt.ylabel('Population')
plt.legend(loc='upper left')
plt.title('Population Dynamics')

# Second subplot: Coherence rho12
plt.subplot(4, 1, 2)
plt.plot(time_points, np.real(solutions[:, 1]), label=r'Re[$\rho_{12}$]')
plt.plot(time_points, np.imag(solutions[:, 1]), label=r'Im[$\rho_{12}$]')
plt.xlabel('Time (μs)')
plt.ylabel(r'Coherence $\rho_{12}$')
plt.legend()
plt.title('Coherence between States 1 and 2')

# Third subplot: Coherence rho13
plt.subplot(4, 1, 3)
plt.plot(time_points, np.real(solutions[:, 2]), label=r'Re[$\rho_{13}$]')
plt.plot(time_points, np.imag(solutions[:, 2]), label=r'Im[$\rho_{13}$]')
plt.xlabel('Time (μs)')
plt.ylabel(r'Coherence $\rho_{13}$')
plt.ylim(-0.001,0.01)
plt.legend()
plt.title('Coherence between States 1 and 3')

# Fourth subplot: Coherence rho23
plt.subplot(4, 1, 4)
plt.plot(time_points, np.real(solutions[:, 4]), label=r'Re[$\rho_{23}$]')
plt.plot(time_points, np.imag(solutions[:, 4]), label=r'Im[$\rho_{23}$]')
plt.xlabel('Time (μs)')
plt.ylabel(r'Coherence $\rho_{23}$')
plt.ylim(-0.001,0.01)
plt.legend()
plt.title('Coherence between States 2 and 3')

# Add parameter values as a text box in one of the subplots (e.g., first subplot)
param_text = (
    f"$\\Omega_p$ = {Om_p} MHz\n"
    f"$\\Omega_c$ = {Om_c} MHz\n"
    f"$\\Delta_p$ = {delta_p} MHz\n"
    f"$\\Delta_c$ = {delta_c} MHz\n"
    f"$\\Gamma_12$ = {Gamma12}\n"
    f"$\\Gamma_21$ = {Gamma21}"
)
plt.subplot(4, 1, 1)
plt.text(
    0.95, 0.95, param_text,
    transform=plt.gca().transAxes,
    fontsize=9,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
)

plt.tight_layout()
plt.show()


