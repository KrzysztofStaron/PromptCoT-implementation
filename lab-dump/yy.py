import numpy as np
from scipy.optimize import curve_fit

# your data
x = np.arange(1, 31)
y = np.array([100,97,94,92,89,86,83,81,78,75,73,71,70,68,67,65,64,63,61,60,59,58,57,56,55,54,53,52,51,50])

# log-logistic model
def log_logistic(x, L, x0, k):
    return L / (1 + (x / x0)**k)

# initial guess can matter
p0 = [100.0, 10.0, 1.0]

# fit
params, cov = curve_fit(log_logistic, x, y, p0=p0, maxfev=10000)
L, x0, k = params

# uncertainties (standard errors)
se = np.sqrt(np.diag(cov))

# goodness-of-fit R^2
y_pred = log_logistic(x, *params)
ss_res = np.sum((y - y_pred)**2)
ss_tot = np.sum((y - np.mean(y))**2)
r2 = 1 - ss_res / ss_tot

print("Fitted parameters:")
print(f" L  = {L:.6f} ± {se[0]:.6f}")
print(f" x0 = {x0:.6f} ± {se[1]:.6f}")
print(f" k  = {k:.6f} ± {se[2]:.6f}")
print(f"R^2 = {r2:.6f}")

# optional: extrapolate and plot
import matplotlib.pyplot as plt
x_ext = np.arange(1, 61)
y_ext = log_logistic(x_ext, *params)

plt.plot(x, y, 'o', label='data')
plt.plot(x_ext, y_ext, '-', label='log-logistic fit')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value (%)')
plt.show()
