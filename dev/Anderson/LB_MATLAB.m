function simulate_OD()
    % simulate_OD.m
    % Main function to simulate Optical Depth (OD) and plot Output Intensities.
    
    % Clear workspace and command window
    clear; clc;
    
    %% Constants
    epsilon_0 = 8.854e-12;   % Vacuum permittivity (F/m)
    hbar = 1.054e-34;        % Reduced Planck's constant (J·s)
    mu_13 = 1.932e-29;       % Dipole moment between states |1> and |3> (C·m)
    mu_23 = 2.53e-29;        % Dipole moment between states |2> and |3> (C·m)
    N = 3.5e10;              % Atomic number density (atoms/m^3)

    Gamma3 = 2 * pi * 6;     % (MHz)
    Gamma31 = (5/9) * Gamma3;  % Decay from level 3 to level 1 (MHz)
    Gamma32 = Gamma3 - Gamma31;  % Decay from level 3 to level 2 (MHz)
    Gamma12 = 0.05 * Gamma3; % Decay rate from level 1 to 2 (MHz)
    Gamma21 = 0.05 * Gamma3; % Decay rate from level 2 to 1 (MHz)
    gamma13 = (Gamma3 + Gamma12) / 2;  % Decoherence rate between levels 1 and 3 (MHz)
    gamma23 = (Gamma3 + Gamma21) / 2;  % Decoherence rate between levels 2 and 3 (MHz)
    gamma12 = (Gamma12 + Gamma21) / 2; % Decoherence rate between ground states (MHz)

    w_1 = 2 * pi * 3e8 / 795e-9;  % Frequency (Hz)
    k_0 = w_1 / 3e8;              % Wave number (1/m)
    L = 0.1;                      % Length in meters
    z_step = 0.00001;             % Step size in meters

    %% Simulation Parameters
    Om_1 = 3;        % Initial Omega_1 value (MHz)
    delta_1 = 0.50;  % Delta_1 value (MHz)
    delta_2 = 0.0;   % Delta_2 value (MHz)

    %% Run Simulation
    fprintf('Starting OD Simulation...\n');
    [Om_2_list_adapted, Om_2_list, Om_1_list, rho11_values, rho22_values, ...
        rho33_values, rho31_values, rho32_values, Om_2_vs_z, Om_1_vs_z, z_array] = ...
        calculate_OD_1(Om_1, delta_1, delta_2, L, z_step, ...
        Gamma3, Gamma31, Gamma32, Gamma12, Gamma21, gamma13, gamma23, gamma12, ...
        epsilon_0, hbar, k_0, N, mu_13, mu_23);

    %% Plot Results
    plot_OD(Om_2_list_adapted, Om_2_list, Om_1_list, Om_2_vs_z, Om_1_vs_z, z_array, Gamma3, Om_1, delta_1, delta_2);
    
    fprintf('Simulation Completed.\n');
end

function [Om_2_list_adapted, Om_2_list, Om_1_list, rho11_values, rho22_values, ...
    rho33_values, rho31_values, rho32_values, Om_2_vs_z, Om_1_vs_z, z_array] = ...
    calculate_OD_1(Om_1, delta_1, delta_2, L, z, ...
    Gamma3, Gamma31, Gamma32, Gamma12, Gamma21, gamma13, gamma23, gamma12, ...
    epsilon_0, hbar, k_0, N, mu_13, mu_23)
    
    % calculate_OD_1.m
    % Calculates the Optical Depth (OD) for given parameters.
    
    % Number of Rabi frequencies and Rabi frequency range
    n_rabi = 500;
    Om_2_list_adapted = linspace(0.005, 12, n_rabi);  % Dimensionless coupling Rabi frequency
    Om_2_list = Om_2_list_adapted * Gamma3;          % Actual Omega_2 in MHz
    Om_1_list = ones(1, n_rabi) * Om_1 * Gamma3;     % Omega_1 array (MHz)

    % Number of z steps and z array
    nz = floor(L / z);          % Number of z steps
    z_array = (0:nz-1) * z;     % Array of z positions

    % Initialize arrays to store populations and coherences
    rho11_values = zeros(nz, n_rabi);
    rho22_values = zeros(nz, n_rabi);
    rho33_values = zeros(nz, n_rabi);
    rho31_values = zeros(nz, n_rabi);
    rho32_values = zeros(nz, n_rabi);

    % Initialize cell arrays to store Rabi frequencies at each z step
    Om_2_vs_z = zeros(nz, n_rabi);
    Om_1_vs_z = zeros(nz, n_rabi);

    % Initialize Rabi frequencies for z=0
    Om_2_vs_z(1, :) = Om_2_list;
    Om_1_vs_z(1, :) = Om_1_list;

    %% Simulation Loop over z
    for i = 1:nz
        % Display progress at quarter intervals
        if i == floor(nz/4) || i == floor(nz/2) || i == floor(3*nz/4)
            fprintf('Processing z step %d/%d\n', i, nz);
        end

        for j = 1:n_rabi
            Om_2_actual = Om_2_list(j);      % Current Omega_2 (MHz)
            Om_1_actual = Om_1_list(j);      % Current Omega_1 (MHz)

            delta_1_actual = delta_1 * Gamma3;
            delta_2_actual = delta_2 * Gamma3;

            % Define the imaginary unit
            I = 1i;

            % Construct the matrix A
            A = [
                Gamma31 + Gamma12, 0, I*Om_1_actual/2, 0, Gamma31 - Gamma21, 0, -I*conj(Om_1_actual)/2, 0;
                0, gamma12 - I*(delta_2_actual - delta_1_actual), I*Om_2_actual/2, 0, 0, 0, 0, -I*conj(Om_1_actual)/2;
                I*conj(Om_1_actual), I*conj(Om_2_actual)/2, gamma13 + I*delta_1_actual, 0, I*conj(Om_1_actual)/2, 0, 0, 0;
                0, 0, 0, gamma12 + I*(delta_2_actual - delta_1_actual), 0, I*Om_1_actual/2, -I*conj(Om_2_actual)/2, 0;
                Gamma32 - Gamma12, 0, 0, 0, Gamma32 + Gamma21, I*Om_2_actual/2, 0, -I*conj(Om_2_actual)/2;
                I*conj(Om_2_actual)/2, 0, 0, I*conj(Om_1_actual)/2, I*conj(Om_2_actual), gamma23 + I*delta_2_actual, 0, 0;
                -I*Om_1_actual, 0, 0, -I*Om_2_actual/2, -I*Om_1_actual/2, 0, gamma13 - I*delta_1_actual, 0;
                -I*Om_2_actual/2, -I*Om_1_actual/2, 0, 0, -I*Om_2_actual, 0, 0, gamma23 - I*delta_2_actual
            ];

            % Construct the vector b
            b = [
                Gamma31;
                0;
                I*conj(Om_1_actual)/2;
                0;
                Gamma32;
                I*conj(Om_2_actual)/2;
                -I*Om_1_actual/2;
                -I*Om_2_actual/2
            ];

            % Solve the linear system A * v = b
            try
                v = A \ b;
            catch
                % Handle singular matrix
                fprintf('Singular matrix at z step %d, Rabi index %d\n', i, j);
                rho11_values(i, j) = NaN;
                rho22_values(i, j) = NaN;
                rho33_values(i, j) = NaN;
                rho31_values(i, j) = NaN;
                rho32_values(i, j) = NaN;
                continue;
            end

            % Extract the solutions
            rho11 = real(v(1));
            rho12 = v(2);
            rho13 = v(3);
            rho21 = v(4);
            rho22 = real(v(5));
            rho23 = v(6);
            rho31 = v(7);
            rho32 = v(8);

            % Compute rho33
            rho33 = 1 - rho11 - rho22;

            % Store the populations
            rho11_values(i, j) = rho11;
            rho22_values(i, j) = rho22;
            rho33_values(i, j) = rho33;
            rho31_values(i, j) = rho31;
            rho32_values(i, j) = rho32;

            % Update the Rabi frequencies
            Om_2_list(j) = Om_2_actual + z * (1i) * (k_0 / (epsilon_0 * hbar)) * N * rho32 * mu_23^2;
            Om_1_list(j) = Om_1_actual + z * (1i) * (k_0 / (epsilon_0 * hbar)) * N * rho31 * mu_13^2;
        end

        % Store Rabi frequencies for this z step
        Om_2_vs_z(i, :) = Om_2_list;
        Om_1_vs_z(i, :) = Om_1_list;
    end
end

function plot_OD(Om_2_list_adapted, Om_2_list, Om_1_list, Om_2_vs_z, Om_1_vs_z, z_array, Gamma3, Om_1, delta_1, delta_2)
    % plot_OD.m
    % Plots the Output Intensities of Omega_1 and Omega_2 vs Initial Omega_2.

    % Calculate Transparency
    Transparency_Omega_1 = abs(Om_1_vs_z(end, :)).^2 ./ abs(Om_1_vs_z(1, :)).^2;
    Transparency_Omega_2 = abs(Om_2_vs_z(end, :)).^2 ./ abs(Om_2_vs_z(1, :)).^2;

    % Calculate Output Intensity based on transparency and initial input power
    Output_Intensity_Omega_1 = Transparency_Omega_1 .* (abs(Om_1_vs_z(1, :)).^2) / (Gamma3^2);
    Output_Intensity_Omega_2 = Transparency_Omega_2 .* (abs(Om_2_vs_z(1, :)).^2) / (Gamma3^2);

    %% Plot Output Intensity of Omega_1 vs Initial Omega_2
    figure('Name', 'Output Intensity of Omega_1', 'NumberTitle', 'off');
    plot(real(Om_2_list_adapted).^2, Output_Intensity_Omega_1, 'b-', 'LineWidth', 1.5);
    xlabel('Initial Intensity 2 ($|\Omega_{2,\mathrm{in}} / \\Gamma_3|^2$)', 'Interpreter', 'latex');
    ylabel('Output Intensity 1 ($|\Omega_{1,\mathrm{out}} / \\Gamma_3|^2$)', 'Interpreter', 'latex');
    title(sprintf('Output Intensity 1 vs $I_{2,\\mathrm{in}} / \\Gamma_3^2$ for $\\Omega_1/\\Gamma_3=%.2f$, $\\Delta_1/\\Gamma_3=%.2f$, $\\Delta_2/\\Gamma_3=%.2f$', Om_1, delta_1, delta_2), 'Interpreter', 'latex');
    grid on;
    legend('Output Intensity $\Omega_1$', 'Interpreter', 'latex', 'Location', 'best');

    %% Plot Output Intensity of Omega_2 vs Initial Omega_2
    figure('Name', 'Output Intensity of Omega_2', 'NumberTitle', 'off');
    plot(real(Om_2_list_adapted).^2, Output_Intensity_Omega_2, 'r-', 'LineWidth', 1.5);
    xlabel('Initial Intensity 2 ($|\Omega_{2,\mathrm{in}} / \\Gamma_3|^2$)', 'Interpreter', 'latex');
    ylabel('Output Intensity 2 ($|\Omega_{2,\mathrm{out}} / \\Gamma_3|^2$)', 'Interpreter', 'latex');
    title(sprintf('Output Intensity 2 vs $I_{2,\\mathrm{in}} / \\Gamma_3^2$ for $\\Omega_1/\\Gamma_3=%.2f$, $\\Delta_1/\\Gamma_3=%.2f$, $\\Delta_2/\\Gamma_3=%.2f$', Om_1, delta_1, delta_2), 'Interpreter', 'latex');
    grid on;
    legend('Output Intensity $\Omega_2$', 'Interpreter', 'latex', 'Location', 'best');
end
