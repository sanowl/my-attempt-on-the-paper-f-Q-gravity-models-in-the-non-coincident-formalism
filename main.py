import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import minimize

# Constants
c = 299792.458  # Speed of light in km/s
G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
Mpc = 3.085677581e22  # Megaparsec in meters

# Hubble function parameterization
def H(z, H0, beta, n):
    return H0 * (z + 1)**n + beta * (1 - (z + 1)**n)

# Non-metricity scalar Q
def Q(z, H0, beta, n, gamma):
    Hz = H(z, H0, beta, n)
    dHz = (H(z+1e-6, H0, beta, n) - H(z-1e-6, H0, beta, n)) / (2e-6)  # Numerical derivative
    return -6 * Hz**2 + 9 * gamma * Hz + 3 * dHz

# f(Q) models
def f_Q_model1(Q, eta, alpha):
    return Q + eta * np.exp(alpha * Q)

def f_Q_model2(Q, eta, alpha):
    return Q + eta * np.log(alpha * Q)

def f_Q_model3(Q, eta, alpha):
    return Q + eta * Q**alpha

def f_Q_model4(Q, eta):
    return Q + eta * Q**(-1)

def f_Q_model5(Q, eta):
    return Q + eta * Q**2

# Friedmann-like equations
def rho(z, H0, beta, n, gamma, f_Q, *f_Q_params):
    Q_val = Q(z, H0, beta, n, gamma)
    F = f_Q(Q_val, *f_Q_params)
    dF = (f_Q(Q_val+1e-6, *f_Q_params) - f_Q(Q_val-1e-6, *f_Q_params)) / (2e-6)  # Numerical derivative
    Hz = H(z, H0, beta, n)
    return (0.5*f_Q(Q_val, *f_Q_params) + (3*Hz**2 - 0.5*Q_val)*F + 1.5*dF*gamma*Q_val) / (8*np.pi*G)

def p(z, H0, beta, n, gamma, f_Q, *f_Q_params):
    Q_val = Q(z, H0, beta, n, gamma)
    F = f_Q(Q_val, *f_Q_params)
    dF = (f_Q(Q_val+1e-6, *f_Q_params) - f_Q(Q_val-1e-6, *f_Q_params)) / (2e-6)  # Numerical derivative
    Hz = H(z, H0, beta, n)
    dHz = (H(z+1e-6, H0, beta, n) - H(z-1e-6, H0, beta, n)) / (2e-6)  # Numerical derivative
    return (-0.5*f_Q(Q_val, *f_Q_params) + (-2*dHz - 3*Hz**2 + 0.5*Q_val)*F + 0.5*dF*(-4*Hz + 3*gamma)*Q_val) / (8*np.pi*G)

# Energy conditions
def NEC(z, H0, beta, n, gamma, f_Q, *f_Q_params):
    return rho(z, H0, beta, n, gamma, f_Q, *f_Q_params) + p(z, H0, beta, n, gamma, f_Q, *f_Q_params)

def SEC(z, H0, beta, n, gamma, f_Q, *f_Q_params):
    return rho(z, H0, beta, n, gamma, f_Q, *f_Q_params) + 3*p(z, H0, beta, n, gamma, f_Q, *f_Q_params)

# Sound speed parameter
def v_s_squared(z, H0, beta, n, gamma, f_Q, *f_Q_params):
    dz = 1e-6
    dp = (p(z+dz, H0, beta, n, gamma, f_Q, *f_Q_params) - p(z-dz, H0, beta, n, gamma, f_Q, *f_Q_params)) / (2*dz)
    drho = (rho(z+dz, H0, beta, n, gamma, f_Q, *f_Q_params) - rho(z-dz, H0, beta, n, gamma, f_Q, *f_Q_params)) / (2*dz)
    return dp / drho

# Cosmographic parameters
def q(z, H0, beta, n):
    Hz = H(z, H0, beta, n)
    dHz = (H(z+1e-6, H0, beta, n) - H(z-1e-6, H0, beta, n)) / (2e-6)  # Numerical derivative
    return -1 + (1 + z) * dHz / Hz

def j(z, H0, beta, n):
    Hz = H(z, H0, beta, n)
    dHz = (H(z+1e-6, H0, beta, n) - H(z-1e-6, H0, beta, n)) / (2e-6)  # Numerical derivative
    d2Hz = (H(z+2e-6, H0, beta, n) - 2*H(z, H0, beta, n) + H(z-2e-6, H0, beta, n)) / (4e-12)  # Second derivative
    return (1 + z)**2 * d2Hz / Hz - (1 + z)**2 * dHz**2 / Hz**2 + 2 * (1 + z) * dHz / Hz + 1

def s(z, H0, beta, n):
    Hz = H(z, H0, beta, n)
    dHz = (H(z+1e-6, H0, beta, n) - H(z-1e-6, H0, beta, n)) / (2e-6)  # Numerical derivative
    d2Hz = (H(z+2e-6, H0, beta, n) - 2*H(z, H0, beta, n) + H(z-2e-6, H0, beta, n)) / (4e-12)  # Second derivative
    d3Hz = (H(z+3e-6, H0, beta, n) - 3*H(z+1e-6, H0, beta, n) + 3*H(z-1e-6, H0, beta, n) - H(z-3e-6, H0, beta, n)) / (8e-18)  # Third derivative
    return (1 + z)**3 * d3Hz / Hz - 3 * (1 + z)**3 * dHz * d2Hz / Hz**2 + 3 * (1 + z)**3 * dHz**3 / Hz**3

# Main analysis
def analyze_model(f_Q, f_Q_params, model_name):
    H0, beta, n = 68, 42, 1.6  # From the paper's results
    gamma = -1 / H(0, H0, beta, n)

    z_range = np.linspace(0, 2, 100)
    
    nec_values = [NEC(z, H0, beta, n, gamma, f_Q, *f_Q_params) for z in z_range]
    sec_values = [SEC(z, H0, beta, n, gamma, f_Q, *f_Q_params) for z in z_range]
    v_s_squared_values = [v_s_squared(z, H0, beta, n, gamma, f_Q, *f_Q_params) for z in z_range]
    
    q_values = [q(z, H0, beta, n) for z in z_range]
    j_values = [j(z, H0, beta, n) for z in z_range]
    s_values = [s(z, H0, beta, n) for z in z_range]

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(z_range, nec_values)
    plt.title('Null Energy Condition')
    plt.xlabel('Redshift (z)')
    plt.ylabel('NEC')
    
    plt.subplot(2, 3, 2)
    plt.plot(z_range, sec_values)
    plt.title('Strong Energy Condition')
    plt.xlabel('Redshift (z)')
    plt.ylabel('SEC')
    
    plt.subplot(2, 3, 3)
    plt.plot(z_range, v_s_squared_values)
    plt.title('Sound Speed Parameter')
    plt.xlabel('Redshift (z)')
    plt.ylabel('v_s^2')
    
    plt.subplot(2, 3, 4)
    plt.plot(z_range, q_values)
    plt.title('Deceleration Parameter')
    plt.xlabel('Redshift (z)')
    plt.ylabel('q(z)')
    
    plt.subplot(2, 3, 5)
    plt.plot(z_range, j_values)
    plt.title('Jerk Parameter')
    plt.xlabel('Redshift (z)')
    plt.ylabel('j(z)')
    
    plt.subplot(2, 3, 6)
    plt.plot(z_range, s_values)
    plt.title('Snap Parameter')
    plt.xlabel('Redshift (z)')
    plt.ylabel('s(z)')
    
    plt.tight_layout()
    plt.suptitle(f'Analysis of {model_name}', fontsize=16)
    plt.show()

# Analyze each model
analyze_model(f_Q_model1, (1, 1), "Model I: f(Q) = Q + η*exp(αQ)")
analyze_model(f_Q_model2, (1, -1), "Model II: f(Q) = Q + η*log(αQ)")
analyze_model(f_Q_model3, (1, -5), "Model III: f(Q) = Q + η*Q^α")
analyze_model(f_Q_model4, (1,), "Model IV: f(Q) = Q + η*Q^(-1)")
analyze_model(f_Q_model5, (1e-6,), "Model V: f(Q) = Q + η*Q^2")