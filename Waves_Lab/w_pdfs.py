import numpy as np

# ==Thermal Waves Experiment==
def sine_with_phase(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/10 * t - phase) + c

def sine_with_phase_15(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/15 * t - phase) + c

def sine_with_phase_17(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/17 * t - phase) + c

def sine_with_phase_20(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/20 * t - phase) + c

def sine_with_phase_30(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/30 * t - phase) + c

def sine_with_phase_60(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/60 * t - phase) + c

def phase_waves(x, m ,c):
    x = np.array(x, dtype=float)
    return m*x + c

def amplitude_waves(x, a, m):
    x = np.array(x, dtype=float)
    if a < 0:
        return np.inf * np.ones_like(x)
    return a*np.exp(m*x)

def beta_from_D(D, omega):
    # complex wavevector
    return np.sqrt(omega / (2*D)) * (1j - 1)

def X_model(x, Cpr, Cpi, D, h_over_k, omega, L):
    x = np.asarray(x, dtype=float)  # <-- ensure array
    C_plus = Cpr + 1j*Cpi
    beta = np.sqrt(omega / (2*D)) * (1j - 1)
    R = (beta + h_over_k) / (-beta + h_over_k)
    term = np.exp(beta*x) - np.exp(2*beta*L) * R * np.exp(-beta*x)
    return C_plus * term  # returns array of same shape as x

def amplitude_model(x, Cpr, Cpi, D, h_over_k, omega, L):
    X = X_model(x, Cpr, Cpi, D, h_over_k, omega, L)
    return np.abs(X)

def phase_model(x, Cpr, Cpi, D, h_over_k, omega, L):
    X = X_model(x, Cpr, Cpi, D, h_over_k, omega, L)
    return np.angle(X)

# ==Electrical Waves Experiment==
def sine_with_phase_elec_20(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 20 * t - phase) + c

def sine_with_phase_elec_578(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 5775 * t - phase) + c

def sine_with_phase_elec_1152(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 11520 * t - phase) + c

def sine_with_phase_elec_1733(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 17330 * t - phase) + c

def sine_with_phase_elec_2316(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 23160 * t - phase) + c

def sine_with_phase_elec_2876(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 28760 * t - phase) + c

def sine_with_phase_elec_3459(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 34590 * t - phase) + c

def sine_with_phase_elec_4026(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 40260 * t - phase) + c

def sine_with_phase_elec_4575(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 45750 * t - phase) + c

def sine_with_phase_elec_5149(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi * 51490 * t - phase) + c

def dispersion_model(k, A):
    return A * np.sin(k / 2)