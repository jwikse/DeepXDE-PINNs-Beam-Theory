# Import libraries
import os
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.integrate as spi
from scipy.interpolate import CubicSpline


# Physical (dimensional) variables
L = 14.0  # Length of the beam (m)
P_point = -6*10**9   # Point load (N)
E = 200.0*10**9  # Young's modulus (Pa)
I = 10.0   # Second moment of area (m^4)
x0 = 5.0      # Location of point load in meters
sigma = 0.03       # Width of Gaussian approximation (smaller = sharper)

# Nondimensionalization 
w_scale = (P_point * L**3) / (E * I) # Characteristic deflection scale
L_hat = L / L  # Normalized length = 1.0
E_hat = E / E  # Normalized modulus = 1.0
I_hat = I / I  # Normalized second moment = 1.0
x0_hat = x0 / L     # Non-dimensional location [0,1]


# Define the geometry
geom = dde.geometry.Interval(0, L_hat)


# Define the non-dimensional PDE
def pde(x, w):
    d2w_dx2 = dde.grad.hessian(w, x)
    d4w_dx4 = dde.grad.hessian(d2w_dx2, x)
    x0_tensor = torch.tensor(x0_hat, dtype=x.dtype)
    
    # Point load as Dirac approximation (Gaussian) in non-dimensional form
    gaussian = torch.exp(-((x - x0_tensor) ** 2) / (2 * sigma ** 2))
    normalization = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * np.pi, dtype=x.dtype)))
    q_hat = gaussian * normalization  # Non-dimensional load distribution

    residual = d4w_dx4 - q_hat  
    
    return residual

# Soft boundary conditions 
def boundary_left(x, on_boundary):

    return on_boundary and np.isclose(x[0], 0)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], L_hat)

def moment_bc(x, w, _):
    return dde.grad.jacobian(dde.grad.jacobian(w, x, i=0), x, i=0)

def shear_bc(x, w, _):
    d2w = dde.grad.jacobian(dde.grad.jacobian(w, x, i=0), x, i=0)
    return dde.grad.jacobian(d2w, x, i=0)

bc_moment_right = dde.OperatorBC(geom, moment_bc, boundary_right)
bc_shear_right = dde.OperatorBC(geom, shear_bc, boundary_right)

bcs = [bc_moment_right, bc_shear_right]

# Combine geometry, PDE, and boundary conditions
data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=1000,
    num_boundary=2,
    solution=None,
    num_test=None
)

# Construct neural network
net = dde.maps.FNN([1] + [10]*3 + [1], "tanh", "Glorot uniform")   #swish and gloro uniform
net.apply_output_transform(lambda x, y: x**2 * y) # Hard boundary condition w(0)=0, and w'(0)=0
model = dde.Model(data, net)

# Compile model with Adam optimizer

model.compile("L-BFGS")
losshistory, train_state = model.train()

# For plotting, define test points in non-dimensional domain
x_test_nd = np.linspace(0, 1, 100)[:, None]
w_pred_nd = model.predict(x_test_nd)



# Analytical solution (non-dimensional) for cantilever beam with point load at x0_hat
w_hat_at_0 = 0.0
slope_hat_at_0 = 0.0  # dw_hat/dx_hat at x_hat=0
w_hat_at_x0 = (P_point * x0**3) / (3 * E * I * w_scale)
w_hat_at_L = (P_point * x0**2 * (3 * L - x0)) / (6 * E * I * w_scale)
slope_hat_at_L = (P_point * x0**2) / (2 * E * I * w_scale) * L
x_key_hat = np.array([0, x0_hat, L_hat])  # [0, x0/L, 1]
w_key_hat = np.array([w_hat_at_0, w_hat_at_x0, w_hat_at_L])
cs = CubicSpline(x_key_hat, w_key_hat, bc_type=((1, slope_hat_at_0), (1, slope_hat_at_L)))
w_reference_nd = cs(x_test_nd.flatten())

# remove # to get dimensional values
x_test_dim = x_test_nd #* L  
w_pred_dim = w_pred_nd #* w_scale  
w_reference = w_reference_nd #* w_scale

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(x_test_dim, -w_pred_dim, 'r-', label='PINN Prediction', linewidth=2)
plt.plot(x_test_dim, -w_reference, 'b--', label='Reference Solution', linewidth=2)
plt.axvline(x0_hat, color='green', linestyle=':', alpha=0.7, label=f'Point Load at x={x0_hat:.2f}m')
plt.xlabel('Length {x_hat}')
plt.ylabel('Deflection {w_hat(x_hat)}')
plt.title(f'Cantilever Beam with Point Load at x={x0_hat:.2f}m: PINN vs Reference')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Right plot: Individual losses (PDE + BCs)
train_loss = np.array(losshistory.loss_train)
plt.semilogy(train_loss[:, 0], label='PDE Loss', linewidth=2)
plt.semilogy(train_loss[:, 1], label='BC Right (Moment)', linewidth=2)
plt.semilogy(train_loss[:, 2], label='BC Right (Shear)', linewidth=2)
plt.xlabel('Epochs (x100)')
plt.ylabel('Loss')
plt.title('Individual Loss Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# Calculate max displacement and relative error
w_max_pred = np.max(w_pred_dim)  # Max because deflection is negative
w_max_exact = np.max(w_reference)
relative_error = abs((w_max_pred - w_max_exact) / w_max_exact) * 100


print(f"\n{'='*60}")
print(f"Max Displacement Results:")
print(f"{'='*60}")
print(f"Predicted max displacement: {w_max_pred:.6e} m")
print(f"Exact max displacement:     {w_max_exact:.6e} m")
print(f"Absolute error:             {abs(w_max_pred - w_max_exact):.6e} m")
print(f"Relative error:             {relative_error:.4f} %")
print(f"{'='*60}\n")

