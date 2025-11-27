import numpy as np

# ==Thermal Waves Experiment==
def sine_with_phase(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/10 * t - phase) + c

def sine_with_phase_15(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/15 * t - phase) + c

def sine_with_phase_20(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/20 * t - phase) + c

def sine_with_phase_30(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/30 * t - phase) + c

def sine_with_phase_60(t, amplitude, phase, c):
    return amplitude * np.sin(2*np.pi/60 * t - phase) + c

def phase_waves(x, m ,c):
    x = np.array(x, dtype=float)
    return m*x + c

def amplitude_waves(x, a, m ,c):
    x = np.array(x, dtype=float)
    if a < 0:
        return np.inf * np.ones_like(x)
    return a*np.exp(m*x) + c