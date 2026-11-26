"""
SEIQ model: neural posterior estimation (sbi) vs. MCMC (emcee)
================================================================
Jonas Kwame Nyador - demo for the PhD interview, Nov 2026

Same model structure as the 2021 MathMods report (S, E, I, Q with recruitment A,
natural death mu, disease death alpha, return from Q to S at rate c), but with a
consistent closed population: A = mu * N, so the disease-free equilibrium is S = N.
The report's transmission value beta * S0 = 0.0006515 * 2000 ~ 1.3/day is kept as a
per-capita contact rate beta_tilde = beta * N.

Observed data: daily NEW confirmed cases = new entries into Q (sigma*I + d*E,
integrated over each day), with negative-binomial reporting noise.

Unknowns: theta = (beta_tilde, sigma, d)
Fixed:    epsilon = 1/6 per day, alpha, mu, c tiny, N = 10,000, E0 = 20, I0 = 10.

NPE here = a mixture-density network (Papamakarios & Murray 2016), written in JAX so it
runs on CPU anywhere. With the sbi package the equivalent is:
    from sbi.inference import NPE; inf = NPE(prior, density_estimator="mdn")
    posterior = inf.build_posterior(inf.append_simulations(theta, x).train())

Run:  pip install numpy scipy emcee jax optax matplotlib
      python seiq_sbi_demo.py        (about 8-10 minutes on a laptop CPU)
"""
import time, json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import nbinom
import emcee
import jax, jax.numpy as jnp, optax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)

# ---------------- model ----------------
N, E0, I0 = 10_000, 20.0, 10.0
eps, alpha, mu, c = 1 / 6, 0.000015, 0.00021, 0.00017
A = mu * N                      # consistent recruitment: DFE is S = N
T = 80                          # days
PHI = 20.0                      # neg-binomial dispersion (var = m + m^2/PHI)
NAMES = [r"$\tilde\beta$ (contact rate)", r"$\sigma$ (I$\to$Q)", r"$d$ (E$\to$Q)"]
LOW, HIGH = np.array([0.3, 0.05, 0.0]), np.array([2.5, 0.9, 0.3])
THETA_TRUE = np.array([1.30, 0.50, 0.05])   # values from the 2021 report


def rhs(t, y, bt, s, d):
    S, E, I, Q, C = y
    inf = bt * S * I / N
    return [A - inf - mu * S + c * Q,
            inf - (eps + mu + d) * E,
            eps * E - (alpha + mu + s) * I,
            s * I + d * E - (alpha + mu + c) * Q,
            s * I + d * E]                     # cumulative confirmed


def expected_cases(theta):
    bt, s, d = theta
    y0 = [N - E0 - I0, E0, I0, 0.0, 0.0]
    sol = solve_ivp(rhs, (0, T), y0, t_eval=np.arange(T + 1), args=(bt, s, d),
                    method="LSODA", rtol=1e-6, atol=1e-6)
    return np.clip(np.diff(sol.y[4]), 1e-6, None)        # daily new confirmed


def simulate(theta, rng=rng):
    m = expected_cases(theta)
    p = PHI / (PHI + m)
    return rng.negative_binomial(PHI, p).astype(float)


def summary(x):
    return np.log1p(x)

# ---------------- observed data ----------------
x_obs = simulate(THETA_TRUE)

# ---------------- NPE: mixture-density network in JAX ----------------
# Train q(z | x) on simulated pairs, with z = logit of theta rescaled to the prior box,
# so every posterior sample automatically respects the prior bounds.
K, D = 8, 3                                   # mixture components, parameter dimension
TRI = np.tril_indices(D)

def to_z(th):   u = np.clip((th - LOW) / (HIGH - LOW), 1e-6, 1 - 1e-6); return np.log(u / (1 - u))
def to_th(z):   return LOW + (HIGH - LOW) / (1 + np.exp(-np.clip(z, -30, 30)))

def init_mlp(key, sizes):
    ps = []
    for m, n in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (m, n)) * np.sqrt(1 / m), jnp.zeros(n)))
    return ps

def mlp(ps, x):
    for W, b in ps[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = ps[-1]
    return x @ W + b

def unpack(out):
    logw = jax.nn.log_softmax(out[:, :K])
    mean = out[:, K:K + K * D].reshape(-1, K, D)
    tri = out[:, K + K * D:].reshape(-1, K, len(TRI[0]))
    U = jnp.zeros((out.shape[0], K, D, D)).at[:, :, TRI[0], TRI[1]].set(tri)
    dg = jnp.diagonal(U, axis1=-2, axis2=-1)
    U = U - jax.vmap(jax.vmap(jnp.diag))(dg) + jax.vmap(jax.vmap(jnp.diag))(jax.nn.softplus(dg) + 1e-3)
    return logw, mean, U                      # U = Cholesky factor of each precision matrix

def log_q(ps, x, z):
    logw, mean, U = unpack(mlp(ps, x))
    y = jnp.einsum("bkji,bkj->bki", U, z[:, None, :] - mean)          # U^T (z - m)
    logdet = jnp.log(jnp.diagonal(U, axis1=-2, axis2=-1)).sum(-1)
    lp = -0.5 * (y ** 2).sum(-1) + logdet - 0.5 * D * np.log(2 * np.pi)
    return jax.scipy.special.logsumexp(logw + lp, axis=1)

def train_npe(x, z, seed=0, epochs=400, batch=256, patience=40):
    xm, xs, zm, zs = x.mean(0), x.std(0) + 1e-6, z.mean(0), z.std(0)
    xn, zn = (x - xm) / xs, (z - zm) / zs
    r = np.random.default_rng(seed); perm = r.permutation(len(x)); nv = len(x) // 10
    va, tr = perm[:nv], perm[nv:]
    ps = init_mlp(jax.random.PRNGKey(seed), [x.shape[1], 256, 256, K + K * D + K * len(TRI[0])])
    opt = optax.adam(5e-4); st = opt.init(ps)
    loss = lambda p, xb, zb: -log_q(p, xb, zb).mean()

    @jax.jit
    def step(p, s, xb, zb):
        l, g = jax.value_and_grad(loss)(p, xb, zb)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, l
    vloss = jax.jit(loss)

    best, best_ps, wait = np.inf, ps, 0
    for ep in range(epochs):
        r.shuffle(tr)
        for i in range(0, len(tr) - batch + 1, batch):
            j = tr[i:i + batch]
            ps, st, _ = step(ps, st, xn[j], zn[j])
        v = float(vloss(ps, xn[va], zn[va]))
        if v < best - 1e-4: best, best_ps, wait = v, ps, 0
        else:
            wait += 1
            if wait >= patience: break
    return dict(ps=best_ps, xm=xm, xs=xs, zm=zm, zs=zs, epochs=ep + 1, val_loss=best)

def sample_npe(model, x, n, rng=rng):
    out = mlp(model["ps"], ((x - model["xm"]) / model["xs"])[None])
    logw, mean, U = [np.asarray(a[0]) for a in unpack(out)]
    k = rng.choice(K, size=n, p=np.exp(logw) / np.exp(logw).sum())
    eps = rng.standard_normal((n, D))
    Linv = np.linalg.inv(np.transpose(U, (0, 2, 1)))                    # (U^T)^-1
    zn = mean[k] + np.einsum("nij,nj->ni", Linv[k], eps)
    return to_th(zn * model["zs"] + model["zm"])

N_SIM = 100_000
t0 = time.time()
theta_train = LOW + (HIGH - LOW) * rng.random((N_SIM, 3))              # prior: uniform box
x_train = np.stack([summary(simulate(th)) for th in theta_train])
t_sim = time.time() - t0
t0 = time.time()
model = train_npe(x_train, to_z(theta_train))
t_train = time.time() - t0
print(f"NPE trained: {model['epochs']} epochs, val loss {model['val_loss']:.3f}")
t0 = time.time()
x_o = summary(x_obs)
npe_samples = sample_npe(model, x_o, 50_000)
t_npe_sample = time.time() - t0

# ---------------- MCMC (exact likelihood) ----------------
def log_post(theta):
    if np.any(theta < LOW) or np.any(theta > HIGH):
        return -np.inf
    m = expected_cases(theta)
    return nbinom.logpmf(x_obs, PHI, PHI / (PHI + m)).sum()

nwalk, nstep, burn = 24, 2500, 800
p0 = THETA_TRUE + 1e-2 * rng.standard_normal((nwalk, 3))
p0 = np.clip(p0, LOW + 1e-3, HIGH - 1e-3)
t0 = time.time()
sampler = emcee.EnsembleSampler(nwalk, 3, log_post)
sampler.run_mcmc(p0, nstep, progress=False)
t_mcmc = time.time() - t0
mcmc_samples = sampler.get_chain(discard=burn, flat=True)
try:
    tau = float(np.max(sampler.get_autocorr_time(discard=burn, quiet=True)))
except Exception:
    tau = float("nan")

# ---------------- calibration check (NPE, amortized) ----------------
n_test, levels = 200, np.linspace(0.05, 0.95, 19)
hits = np.zeros((len(levels), 3))
t0 = time.time()
for _ in range(n_test):
    th = LOW + (HIGH - LOW) * rng.random(3)
    smp = sample_npe(model, summary(simulate(th)), 1000)
    for i, lv in enumerate(levels):
        lo, hi = np.quantile(smp, [(1 - lv) / 2, (1 + lv) / 2], axis=0)
        hits[i] += (th >= lo) & (th <= hi)
t_cal = time.time() - t0
coverage = hits / n_test

# ---------------- results ----------------
def summ(s):
    return {"mean": s.mean(0).round(3).tolist(),
            "q05": np.quantile(s, 0.05, 0).round(3).tolist(),
            "q95": np.quantile(s, 0.95, 0).round(3).tolist()}

res = {"theta_true": THETA_TRUE.tolist(), "n_sim": N_SIM, "npe": summ(npe_samples), "mcmc": summ(mcmc_samples),
       "corr_npe": np.corrcoef(npe_samples.T).round(2).tolist(),
       "corr_mcmc": np.corrcoef(mcmc_samples.T).round(2).tolist(),
       "R_true": float(THETA_TRUE[0] * eps / ((eps + mu + THETA_TRUE[2]) * (alpha + mu + THETA_TRUE[1]))),
       "seconds": {"simulate_train": round(t_sim, 1), "train_npe": round(t_train, 1),
                   "npe_sample_5000": round(t_npe_sample, 2), "mcmc": round(t_mcmc, 1),
                   "npe_200_new_datasets": round(t_cal, 1)},
       "mcmc_lik_evals": nwalk * nstep, "mcmc_autocorr_max": round(tau, 1),
       "mcmc_accept": float(np.mean(sampler.acceptance_fraction).round(2))}
json.dump(res, open("results.json", "w"), indent=2)
np.save("npe_samples.npy", npe_samples); np.save("mcmc_samples.npy", mcmc_samples)
np.save("x_obs.npy", x_obs); np.save("coverage.npy", coverage)
print(json.dumps(res, indent=2))

# ---------------- figures ----------------
INK, TEAL, CORAL, GREY = "#14213D", "#0F7173", "#E76F51", "#8A99A6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})

# 1. marginals NPE vs MCMC
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
for k, ax in enumerate(axs):
    bins = np.linspace(min(npe_samples[:, k].min(), mcmc_samples[:, k].min()),
                       max(npe_samples[:, k].max(), mcmc_samples[:, k].max()), 50)
    ax.hist(mcmc_samples[:, k], bins=bins, density=True, color=GREY, alpha=0.55, label="MCMC (emcee)")
    ax.hist(npe_samples[:, k], bins=bins, density=True, histtype="step", lw=2.2, color=TEAL, label="NPE (neural)")
    ax.axvline(THETA_TRUE[k], color=CORAL, lw=2.2, ls="--", label="true value")
    ax.set_xlabel(NAMES[k]); ax.set_yticks([])
axs[0].legend(frameon=False, fontsize=11, loc="upper left")
fig.tight_layout(); fig.savefig("fig_posteriors.png", dpi=200, transparent=True); plt.close(fig)

# 2. data + posterior predictive
fig, ax = plt.subplots(figsize=(7.5, 4.2))
idx = rng.choice(len(npe_samples), 300, replace=False)
preds = np.array([expected_cases(npe_samples[i]) for i in idx])
days = np.arange(1, T + 1)
ax.fill_between(days, *np.quantile(preds, [0.05, 0.95], 0), color=TEAL, alpha=0.25, lw=0,
                label="NPE posterior, 90% band")
ax.plot(days, np.median(preds, 0), color=TEAL, lw=2.2, label="posterior median")
ax.scatter(days, x_obs, s=16, color=INK, zorder=3, label="observed (synthetic, noisy)")
ax.plot(days, expected_cases(THETA_TRUE), color=CORAL, lw=1.6, ls="--", label="truth")
ax.set_xlabel("day"); ax.set_ylabel("new confirmed cases / day")
ax.legend(frameon=False, fontsize=11)
fig.tight_layout(); fig.savefig("fig_fit.png", dpi=200, transparent=True); plt.close(fig)

# 3. calibration
fig, ax = plt.subplots(figsize=(4.4, 4.2))
ax.plot([0, 1], [0, 1], color=GREY, lw=1.2, ls=":")
for k, col in zip(range(3), [TEAL, CORAL, INK]):
    ax.plot(levels, coverage[:, k], "o-", ms=4, color=col, lw=1.8, label=NAMES[k].split(" (")[0])
ax.set_xlabel("credible level"); ax.set_ylabel("empirical coverage")
ax.legend(frameon=False, fontsize=11)
fig.tight_layout(); fig.savefig("fig_coverage.png", dpi=200, transparent=True); plt.close(fig)

# 4. joint (beta_tilde, sigma)
fig, ax = plt.subplots(figsize=(4.4, 4.2))
ax.scatter(mcmc_samples[::10, 0], mcmc_samples[::10, 1], s=4, color=GREY, alpha=0.4, label="MCMC")
ax.scatter(npe_samples[:1500, 0], npe_samples[:1500, 1], s=4, color=TEAL, alpha=0.4, label="NPE")
ax.scatter(*THETA_TRUE[:2], marker="*", s=220, color=CORAL, zorder=3, label="truth")
ax.set_xlabel(NAMES[0].split(" (")[0]); ax.set_ylabel(NAMES[1].split(" (")[0])
ax.legend(frameon=False, fontsize=11, markerscale=2)
fig.tight_layout(); fig.savefig("fig_joint.png", dpi=200, transparent=True); plt.close(fig)
print("done")
