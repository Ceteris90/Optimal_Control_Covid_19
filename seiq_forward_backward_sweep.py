"""
Optimal control of the SEIQ model with Pontryagin's maximum principle,
solved by a forward-backward sweep (RK4 forwards for the state, RK4 backwards
for the adjoint, projected control update with relaxation).

Model and controls as in the 2021 MathMods report:
  u1 = vaccination (removes S and E),  u2 = treatment (removes I)
  J  = int_0^T ( A1 E + A2 I + w1 u1^2 + w2 u2^2 ) dt,   0 <= u1, u2 <= 1

Run:  python seiq_forward_backward_sweep.py      (writes figures/ and prints results)
"""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

# parameters from the 2021 report (Giordano et al. 2020, WHO, Index Mundi)
A, beta, eps, alpha, sigma, mu, d, c = 0.001, 0.0006515, 0.1667, 0.000015, 0.50, 0.00021, 0.050, 0.00017
T, N = 100, 4000; t = np.linspace(0, T, N + 1); h = t[1] - t[0]
x0 = np.array([2000., 1000., 70., 20.])          # S0, E0, I0, Q0
A1, A2, w1, w2 = 1.0, 1.0, 1000.0, 1000.0        # illustrative weights

def f(x, u1, u2):                                 # state equations
    S, E, I, Q = x
    return np.array([A - beta*S*I - mu*S + c*Q - u1*S,
                     beta*S*I - (eps + mu + d)*E - u1*E,
                     eps*E - (alpha + mu + sigma)*I - u2*I,
                     sigma*I + d*E - (alpha + mu + c)*Q])

def g(x, l, u1, u2):                              # adjoint equations  l' = -dH/dx
    S, E, I, Q = x; l1, l2, l3, l4 = l
    dHdS = -(beta*I + mu + u1)*l1 + beta*I*l2
    dHdE = A1 - (eps + mu + d + u1)*l2 + eps*l3 + d*l4
    dHdI = A2 - beta*S*l1 + beta*S*l2 - (alpha + mu + sigma + u2)*l3 + sigma*l4
    dHdQ = c*l1 - (alpha + mu + c)*l4
    return -np.array([dHdS, dHdE, dHdI, dHdQ])

def fwd(u1, u2):
    X = np.zeros((N + 1, 4)); X[0] = x0
    for k in range(N):
        a = (u1[k] + u1[k+1])/2; b = (u2[k] + u2[k+1])/2
        k1 = f(X[k], u1[k], u2[k]); k2 = f(X[k] + h/2*k1, a, b)
        k3 = f(X[k] + h/2*k2, a, b); k4 = f(X[k] + h*k3, u1[k+1], u2[k+1])
        X[k+1] = X[k] + h/6*(k1 + 2*k2 + 2*k3 + k4)
    return X

def bwd(X, u1, u2):                               # lambda(T) = 0 (no terminal cost)
    L = np.zeros((N + 1, 4))
    for k in range(N, 0, -1):
        xm = (X[k] + X[k-1])/2; a = (u1[k] + u1[k-1])/2; b = (u2[k] + u2[k-1])/2
        k1 = g(X[k], L[k], u1[k], u2[k]); k2 = g(xm, L[k] - h/2*k1, a, b)
        k3 = g(xm, L[k] - h/2*k2, a, b); k4 = g(X[k-1], L[k] - h*k3, u1[k-1], u2[k-1])
        L[k-1] = L[k] - h/6*(k1 + 2*k2 + 2*k3 + k4)
    return L

J = lambda X, u1, u2: np.trapezoid(A1*X[:, 1] + A2*X[:, 2] + w1*u1**2 + w2*u2**2, t)
zero = np.zeros(N + 1)

def solve(use_u1=True, use_u2=True, tol=1e-7):
    u1, u2 = zero.copy(), zero.copy()
    for it in range(1000):
        X = fwd(u1, u2); L = bwd(X, u1, u2)
        n1 = np.clip((L[:, 0]*X[:, 0] + L[:, 1]*X[:, 1])/(2*w1), 0, 1) if use_u1 else zero
        n2 = np.clip(L[:, 2]*X[:, 2]/(2*w2), 0, 1) if use_u2 else zero
        err = max(abs(n1 - u1).max(), abs(n2 - u2).max())
        u1, u2 = 0.5*u1 + 0.5*n1, 0.5*u2 + 0.5*n2       # relaxation
        if err < tol: break
    X = fwd(u1, u2); return X, u1, u2, J(X, u1, u2), it

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    X0 = fwd(zero, zero); J0 = J(X0, zero, zero)
    Xt, *_ , Jt, _ = solve(use_u1=False)
    Xv, *_ , Jv, _ = solve(use_u2=False)
    X, u1, u2, Jc, it = solve()
    den = (eps + mu + d)*(alpha + mu + sigma)
    print(f"R0 = {beta*(A/mu)*eps/den:.4f},  R_eff(0) = {beta*x0[0]*eps/den:.2f},  sweep iterations = {it}")
    for name, XX, JJ in [("none", X0, J0), ("treatment", Xt, Jt), ("vaccination", Xv, Jv), ("both", X, Jc)]:
        print(f"{name:12s} peak I = {XX[:, 2].max():6.1f}   cost J = {JJ:8.0f}   infected person-days = {np.trapezoid(XX[:, 2], t):6.0f}")

    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
    C = {"S": "#2E6FB7", "E": "#E0A030", "I": "#C0392B", "Q": "#3C8D5A"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, k in enumerate("SEIQ"): ax.plot(t, X0[:, i], c=C[k], lw=2.4, label=k)
    ax.set_xlabel("time (days)"); ax.set_ylabel("individuals"); ax.set_xlim(0, 60); ax.legend(frameon=False, ncol=4)
    ax.set_title("SEIQ dynamics without control"); fig.tight_layout(); fig.savefig("figures/uncontrolled.png", dpi=160)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for lab, XX, sty, col in [("no control", X0, "--", "#888"), ("treatment only", Xt, ":", "#8E6CB8"),
                              ("vaccination only", Xv, "-.", "#E0A030"), ("both controls", X, "-", "#1F3A5F")]:
        ax[0].plot(t, XX[:, 2], sty, c=col, lw=2.4, label=lab); ax[1].plot(t, XX[:, 1], sty, c=col, lw=2.4)
    ax[0].set_title("Infected I(t)"); ax[1].set_title("Exposed E(t)")
    for a in ax: a.set_xlabel("time (days)"); a.set_xlim(0, 50)
    ax[0].legend(frameon=False); fig.tight_layout(); fig.savefig("figures/scenarios.png", dpi=160)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(t, u1, c="#2E6FB7", lw=2.6, label="u1: vaccination (S, E)"); ax.plot(t, u2, c="#C0392B", lw=2.6, label="u2: treatment (I)")
    ax.set_xlim(0, 40); ax.set_ylim(-0.02, 1.05); ax.set_xlabel("time (days)"); ax.set_ylabel("control intensity")
    ax.set_title("Optimal controls"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig("figures/controls.png", dpi=160)
