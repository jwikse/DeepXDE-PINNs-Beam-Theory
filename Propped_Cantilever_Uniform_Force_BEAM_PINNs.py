#import libraries
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt

# Physical (dimensional) variables
L = 14.0  # Length of the beam (m)
q = -6.0   # Uniform force (N/m)
E = 200.0  # Young's modulus (MPa)
I = 10.0   # Second moment of area (m^4)

# Nondimensionalization 
w_scale = (q * L**4) / (E * I) # Characteristic deflection scale
q_hat = q * L**4 / (E * I * w_scale) # Nondimensional uniform load = 1.0
L_hat = L / L  # Normalized length = 1.0
E_hat = E / E  # Normalized modulus = 1.0
I_hat = I / I  # Normalized second moment = 1.0

# Define the geometry (non-dimensional)
geom = dde.geometry.Interval(0, L_hat)

# Define nondimensional PDE: EI*d⁴w/dx⁴ = q(x) 
def pde(x, w):
    d2w_dx2 = dde.grad.hessian(w, x)
    d4w_dx4 = dde.grad.hessian(d2w_dx2, x)
    return E_hat*I_hat*d4w_dx4 - q_hat

# Soft boundary conditions: w'(0)=0, w''(L)=0
def bc_left(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], 0)

def bc_right(x, on_boundary):
    return on_boundary and dde.utils.isclose(x[0], L_hat)

def moment_bc(x, w, _):
    return dde.grad.hessian(w, x)

bc_rotation_left = dde.NeumannBC(geom, lambda x: 0.0, bc_left)
bc_moment_right = dde.OperatorBC(geom, moment_bc, bc_right)

bcs = [bc_rotation_left, bc_moment_right]


# Model setup
data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=1000,
    num_boundary=2,
    solution=None,
    num_test=None
)
net = dde.maps.FNN([1] + [10] * 3 + [1], "tanh", "Glorot uniform")
net.apply_output_transform(lambda x, y: x * (L_hat - x) * y) # Hard boundary condition w(0)=w(1)=0
model = dde.Model(data, net)


# Train Model
model.compile("L-BFGS")
losshistory, train_state = model.train()


# Analytical solution for propped cantilever beam under uniform load (non-dimensional)
def beam_analytical_nondim(x):
    return (x**2 / 48) * (3 - 5 * x + 2 * x**2)*(q_hat / (E_hat * I_hat))


# Predict in non-dimensional domain
x_values = np.linspace(0, 1, 100)
w_pred_nondim = model.predict(x_values.reshape(-1, 1))
w_exact_nondim = beam_analytical_nondim(x_values)

# remove # to get dimensional values
x_dim = x_values #* L
w_pred_dim = w_pred_nondim #* w_scale
w_exact_dim = w_exact_nondim #* w_scale

# Plot dimensional results
plt.plot(x_dim, -w_pred_dim, label="Predicted", color='blue')
plt.plot(x_dim, -w_exact_dim, label="Exact", linestyle='dashed', color='red')
plt.xlabel("Length {x_hat}")
plt.ylabel("Deflection {w_hat(x_hat)}")
plt.title('Propped Cantilever Beam Deflection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Right plot: Individual losses (PDE + BCs)
train_loss = np.array(losshistory.loss_train)
plt.semilogy(train_loss[:, 0], label='PDE Loss', linewidth=2)
plt.semilogy(train_loss[:, 1], label='BC Left (rotation)', linewidth=2)
plt.semilogy(train_loss[:, 2], label='BC Right (Moment)', linewidth=2)
plt.xlabel('Epochs (x100)')
plt.ylabel('Loss')
plt.title('Individual Loss Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Calculate max displacement and relative error
w_max_pred = np.max(w_pred_dim)  # Min because deflection is negative
w_max_exact = np.max(w_exact_dim)
relative_error = abs((w_max_pred - w_max_exact) / w_max_exact) * 100

print(f"\n{'='*60}")
print(f"Max Displacement Results:")
print(f"{'='*60}")
print(f"Predicted max displacement: {w_max_pred:.6e} m")
print(f"Exact max displacement:     {w_max_exact:.6e} m")
print(f"Absolute error:             {abs(w_max_pred - w_max_exact):.6e} m")
print(f"Relative error:             {relative_error:.4f} %")
print(f"{'='*60}\n")

