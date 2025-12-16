# Import libraries
import os
os.environ['DDE_BACKEND'] = 'pytorch'
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import torch

# Physical (dimensional) variables
L = 14.0  # Length of the beam (m)
P_point = -10.0      # Magnitude of point force
E = 200.0  # Young's modulus (MPa)
I = 10.0   # Second moment of area (m^4)
x0 = L/L        # Location of point load (at free end)
sigma = 0.02       # Width of Gaussian approximation (smaller = sharper)

# Nondimensionalization 
w_scale = (P_point * L**3) / (E * I) # Characteristic deflection scale
q_hat = P_point * L**4 / (E * I * w_scale) # Nondimensional uniform load = 1.0
L_hat = L / L  # Normalized length = 1.0
E_hat = E / E  # Normalized modulus = 1.0
I_hat = I / I  # Normalized second moment = 1.0

# Define the geometry
geom = dde.geometry.Interval(0, L_hat)


# Define the non-dimensional PDE
def pde(x, w):
    d2w_dx2 = dde.grad.hessian(w, x)
    d4w_dx4 = dde.grad.hessian(d2w_dx2, x)
    x0_tensor = torch.tensor(x0, dtype=x.dtype)
    
    # Point load as Dirac approximation (Gaussian) in non-dimensional form
    gaussian = torch.exp(-((x - x0_tensor) ** 2) / (2 * sigma ** 2))
    normalization = 1.0 / (sigma * torch.sqrt(torch.tensor(2.0 * np.pi, dtype=x.dtype)))
    q_hat = gaussian * normalization  # Non-dimensional load distribution

    residual = d4w_dx4 - q_hat * 2  # factor of 2 since half is outside domain
    
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
net = dde.maps.FNN([1] + [10] * 3 + [1], "tanh", "Glorot uniform")  
net.apply_output_transform(lambda x, y: x**2 * y) # Hard boundary condition w(0)=0
model = dde.Model(data, net)

model.compile("L-BFGS")
losshistory, train_state = model.train()

# Analytical solution (non-dimensional)
def beam_analytical_nondim(x):
    return (x**2 / 6) * (3 - x)

# For plotting, define test points in non-dimensional domain
x_test = np.linspace(0, 1, 200)[:, None]
w_analytical_nd = beam_analytical_nondim(x_test)

# After prediction, convert to dimensional values for plotting
w_pred_nd = model.predict(x_test)

# remove # to get dimensional values
x_dim = x_test #* L
w_pred_dim = w_pred_nd #* w_scale
w_analytical_dim = w_analytical_nd #* w_scale

# Plot dimensional results
plt.figure(figsize=(10, 6))
plt.plot(x_dim, -w_pred_dim, 'r-', label='PINN Prediction', linewidth=2)
plt.plot(x_dim, -w_analytical_dim, 'b--', label='Analytical Solution', linewidth=2)
plt.xlabel('Position along beam {x_hat}')
plt.ylabel('Deflection {w_hat(x_hat)}')
plt.title('Cantilever Beam with Point Load')
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
w_max_pred = np.max(w_pred_dim)  # Min because deflection is negative
w_max_exact = np.max(w_analytical_dim)
relative_error = abs((w_max_pred - w_max_exact) / w_max_exact) * 100

print(f"\n{'='*60}")
print(f"Max Displacement Results:")
print(f"{'='*60}")
print(f"Predicted max displacement: {w_max_pred:.6e} m")
print(f"Exact max displacement:     {w_max_exact:.6e} m")
print(f"Absolute error:             {abs(w_max_pred - w_max_exact):.6e} m")
print(f"Relative error:             {relative_error:.4f} %")
print(f"{'='*60}\n")