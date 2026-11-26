# Reproduces the demo's observed data and MCMC posterior (same seed, same model)
import numpy as np, time
from scipy.integrate import solve_ivp
from scipy.stats import nbinom
import emcee
rng = np.random.default_rng(1)
N, E0, I0 = 10_000, 20.0, 10.0
eps, alpha, mu, c = 1/6, 0.000015, 0.00021, 0.00017
A = mu*N; T = 80; PHI = 20.0
LOW, HIGH = np.array([0.3,0.05,0.0]), np.array([2.5,0.9,0.3])
THETA_TRUE = np.array([1.30,0.50,0.05])
def rhs(t,y,bt,s,d):
    S,E,I,Q,C=y; inf=bt*S*I/N
    return [A-inf-mu*S+c*Q, inf-(eps+mu+d)*E, eps*E-(alpha+mu+s)*I, s*I+d*E-(alpha+mu+c)*Q, s*I+d*E]
def expected_cases(th):
    sol=solve_ivp(rhs,(0,T),[N-E0-I0,E0,I0,0,0],t_eval=np.arange(T+1),args=tuple(th),method="LSODA",rtol=1e-6,atol=1e-6)
    return np.clip(np.diff(sol.y[4]),1e-6,None)
def simulate(th):
    m=expected_cases(th); return rng.negative_binomial(PHI,PHI/(PHI+m)).astype(float)
x_obs=simulate(THETA_TRUE)
def log_post(th):
    if np.any(th<LOW) or np.any(th>HIGH): return -np.inf
    m=expected_cases(th); return nbinom.logpmf(x_obs,PHI,PHI/(PHI+m)).sum()
nwalk,nstep,burn=24,2500,800
p0=np.clip(THETA_TRUE+1e-2*rng.standard_normal((nwalk,3)),LOW+1e-3,HIGH-1e-3)
t0=time.time(); sm=emcee.EnsembleSampler(nwalk,3,log_post); sm.run_mcmc(p0,nstep,progress=False)
ch=sm.get_chain(discard=burn,flat=True)
print('mcmc s',round(time.time()-t0),'mean',ch.mean(0).round(3),'q05',np.quantile(ch,.05,0).round(3),'q95',np.quantile(ch,.95,0).round(3),'corr',np.corrcoef(ch.T)[0,1].round(2))
np.save('mcmc_samples.npy',ch); np.save('x_obs.npy',x_obs)
