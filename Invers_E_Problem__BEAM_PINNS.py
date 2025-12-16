# Import libraries
import numpy as np
import deepxde as dde
import matplotlib.pyplot as plt


# Physical (dimensional) variables
L = 14.0           # Length of the beam (m)
q = -6.0           # uniform load (N/m)
I = 10.0           # second moment of area (m^4)
E_true = 186.0     # true Young’s modulus (MPa)
E_ref = 186.0      # reference modulus for nondimensionalization (MPa)

# Nondimensionalization
w_scale = (q * L**4) / (E_ref * I) # Characteristic deflection scale
L_hat = L / L  # Normalized length = 1.0 
I_hat = I / I  # Normalized second moment = 1.0 
q_hat = q / q  # Normalized load = 1.0 


def w_hat_analytic(x):  
    return (x**4)/24 - (x**3)/6 + (x**2)/4


def w_phys_analytic(x):
    return w_hat_analytic(x/L) * w_scale


geom = dde.geometry.Interval(0.0, L_hat)


E_hat = dde.Variable(1.15)   

def pde_hat(x, y):
    dy_xx = dde.grad.hessian(y, x)
    dy_xxxx = dde.grad.hessian(dy_xx, x)
    return E_hat *I_hat* dy_xxxx - q_hat


# Soft boundary conditions
def left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

def right(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1.0)

# slope BC: w'(0) = 0
def slope_left(x, w, _):
    return dde.grad.jacobian(w, x, i=0)

# moment BC: w''(1) = 0
def moment_right(x, w, _):
    return dde.grad.hessian(w, x)

# shear BC: w'''(1) = 0
def shear_right(x, w, _):
    w_xx = dde.grad.hessian(w, x)
    return dde.grad.jacobian(w_xx, x, i=0)


bc_slope_left  = dde.OperatorBC(geom, slope_left, left)
bc_moment_right = dde.OperatorBC(geom, moment_right, right)
bc_shear_right  = dde.OperatorBC(geom, shear_right, right)


#non dimensional training data from physical analytic solution
x_data_phys = np.linspace(0, L, 80).reshape(-1, 1)
x_data_hat  = (x_data_phys / L).astype(np.float32)
w_data_phys = w_phys_analytic(x_data_phys)
noise_level = 0.02
w_data_phys += noise_level * np.abs(w_data_phys) * np.random.randn(*w_data_phys.shape)

w_data_hat  = (w_data_phys / w_scale).astype(np.float32)

# data loss
data_loss = dde.PointSetBC(x_data_hat, w_data_hat, component=0)


bcs = [ bc_slope_left, bc_moment_right, bc_shear_right, data_loss,]


data = dde.data.PDE(
    geom,
    pde_hat,
    bcs,
    num_domain=1000,      
    num_boundary=2,
)


net = dde.maps.FNN([1] + [10]*3 + [1], "tanh", "Glorot uniform",)
net.apply_output_transform(lambda x, y: x * y) # Hard boundary condition w(0)=0
model = dde.Model(data, net)

model.compile("L-BFGS", external_trainable_variables=[E_hat])

# Simple callback that just prints
class PrintVariable(dde.callbacks.Callback):
    def __init__(self, var, E_ref, period=200):
        super().__init__()
        self.var = var
        self.E_ref = E_ref
        self.period = period
        self.model = None  # Initialize model attribute
    
    def on_epoch_end(self):
        if self.model.train_state.step % self.period == 0:
            val = float(self.var.detach().cpu().numpy())
            print(f"E_hat = {val:.6f}, E_pred = {val * self.E_ref:.4f}")


variable = dde.callbacks.VariableValue(E_hat, period=200, filename="variables.dat")
print_var = PrintVariable(E_hat, E_ref, period=200)
losshistory, train_state = model.train(
    epochs=50000,
    display_every=200,
    callbacks=[variable, print_var],
)


Ehat_pred = float(E_hat.detach().cpu().numpy())
E_pred = Ehat_pred * E_ref

print("Predicted E:", E_pred)
print("True E:", E_true)
print("Error [%]:", abs(E_pred - E_true)/E_true*100)


lines = open("variables.dat", "r").readlines()
E_history = np.array([np.float32(line.split()[1].strip('[]')) for line in lines])
E_history_phys = E_history * E_ref

plt.figure(figsize=(10, 6))
plt.plot(E_history_phys, label='Predicted E', linewidth=2)
plt.axhline(y=E_true, color='r', linestyle='--', label=f'True E = {E_true}', linewidth=2)
plt.xlabel('Epoch (x200)')
plt.ylabel('Young\'s Modulus [GPa]')
plt.title('Evolution of Predicted Young\'s Modulus')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


x_test = np.linspace(0, L, 200).reshape(-1, 1)
x_test_hat = x_test / L
w_pred_hat = model.predict(x_test_hat)
w_pred = w_pred_hat * w_scale

w_exact = w_phys_analytic(x_test)

plt.figure(figsize=(8,5))
plt.plot(x_test, w_pred, label="PINN prediction")
plt.plot(x_test, w_exact, '--', label="Analytical")
plt.xlabel("x [m]")
plt.ylabel("w [m]")
plt.legend()
plt.grid(True)
plt.show()
