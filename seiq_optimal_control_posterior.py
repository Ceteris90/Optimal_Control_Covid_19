"""Optimal control (Pontryagin, forward-backward sweep) for each posterior sample.
Consistent SEIQ model from seiq_sbi_demo.py; controls as in the 2021 report:
u1 = vaccination (removes S and E), u2 = treatment (removes I).
J = int_0^T (A1 E + A2 I + w1 u1^2 + w2 u2^2) dt, 0 <= u <= 1. Vectorised over samples."""
import numpy as np, json, sys
N,E0,I0=10_000,20.,10.; eps,alpha,mu,c=1/6,0.000015,0.00021,0.00017; A=mu*N
T,h=100.,0.05; M=int(T/h); t=np.linspace(0,T,M+1)
A1=A2=1.0; w1=w2=float(sys.argv[1]) if len(sys.argv)>1 else 1000.0
def f(x,u1,u2,p):
    S,E,I,Q=x; b,s,d=p; inf=b*S*I/N
    return np.stack([A-inf-mu*S+c*Q-u1*S, inf-(eps+mu+d)*E-u1*E, eps*E-(alpha+mu+s)*I-u2*I, s*I+d*E-(alpha+mu+c)*Q])
def g(x,l,u1,u2,p):
    S,E,I,Q=x; l1,l2,l3,l4=l; b,s,d=p
    return -np.stack([(-b*I/N-mu-u1)*l1+b*I/N*l2,
                      A1-(eps+mu+d+u1)*l2+eps*l3+d*l4,
                      A2-b*S/N*l1+b*S/N*l2-(alpha+mu+s+u2)*l3+s*l4,
                      c*l1-(alpha+mu+c)*l4])
def fwd(u1,u2,p):
    n=p.shape[1]; X=np.zeros((M+1,4,n)); X[0]=np.array([N-E0-I0,E0,I0,0.])[:,None]
    for k in range(M):
        a=(u1[k]+u1[k+1])/2; b=(u2[k]+u2[k+1])/2; x=X[k]
        k1=f(x,u1[k],u2[k],p); k2=f(x+h/2*k1,a,b,p); k3=f(x+h/2*k2,a,b,p); k4=f(x+h*k3,u1[k+1],u2[k+1],p)
        X[k+1]=x+h/6*(k1+2*k2+2*k3+k4)
    return X
def bwd(X,u1,u2,p):
    L=np.zeros_like(X)
    for k in range(M,0,-1):
        xm=(X[k]+X[k-1])/2; a=(u1[k]+u1[k-1])/2; b=(u2[k]+u2[k-1])/2; l=L[k]
        k1=g(X[k],l,u1[k],u2[k],p); k2=g(xm,l-h/2*k1,a,b,p); k3=g(xm,l-h/2*k2,a,b,p); k4=g(X[k-1],l-h*k3,u1[k-1],u2[k-1],p)
        L[k-1]=l-h/6*(k1+2*k2+2*k3+k4)
    return L
def cost(X,u1,u2): return np.trapezoid(A1*X[:,1]+A2*X[:,2]+w1*u1**2+w2*u2**2,t,axis=0)
def solve(p,iters=1500,tol=1e-5):
    n=p.shape[1]; u1=np.zeros((M+1,n)); u2=np.zeros((M+1,n)); r=np.full(n,0.5); prev=np.full(n,np.inf)
    for it in range(iters):
        X=fwd(u1,u2,p); L=bwd(X,u1,u2,p)
        n1=np.clip((L[:,0]*X[:,0]+L[:,1]*X[:,1])/(2*w1),0,1); n2=np.clip(L[:,2]*X[:,2]/(2*w2),0,1)
        err=np.maximum(np.max(np.abs(n1-u1),axis=0),np.max(np.abs(n2-u2),axis=0))
        r=np.where(err>prev*0.999,np.maximum(r*0.6,0.01),np.minimum(r*1.05,0.5)); prev=err
        u1=(1-r)*u1+r*n1; u2=(1-r)*u2+r*n2
        if err.max()<tol: break
    return u1,u2,fwd(u1,u2,p),err,it
rng=np.random.default_rng(7)
ch=np.load('mcmc_samples.npy'); idx=rng.choice(len(ch),300,replace=False)
P=np.column_stack([ch[idx].T, np.array([1.30,0.50,0.05])[:,None]])  # last column = truth
u1,u2,X,err,it=solve(P)
z=np.zeros_like(u1); X0=fwd(z,z,P)
J0=cost(X0,z,z); Jc=cost(X,u1,u2)
def first_below(u,thr=0.01):
    out=[]
    for j in range(u.shape[1]):
        k=np.argmax(u[:,j]); after=np.where(u[k:,j]<thr)[0]; out.append(t[k+after[0]] if len(after) else T)
    return np.array(out)
tpk=t[np.argmax(u1,axis=0)]; toff=first_below(u1)
cum=np.cumsum(u1,axis=0); frac20=cum[int(20/h)]/cum[-1]
ok=(u1[0]>=0.99)&(tpk<=1.0)
# robustness: apply the policy optimised for the truth to every sample
ut1=np.repeat(u1[:,-1:],P.shape[1],1); ut2=np.repeat(u2[:,-1:],P.shape[1],1)
Xr=fwd(ut1,ut2,P); Jr=cost(Xr,ut1,ut2); regret=(Jr-Jc)/Jc
S=slice(0,-1)
res=dict(weights=w1,iters=int(it),max_err=float(err.max()),n=int(P.shape[1]-1),
 share_early_and_hard=float(ok[S].mean()),
 u1_at_0_median=float(np.median(u1[0,S])),
 u1_switch_off_day=dict(q05=float(np.quantile(toff[S],.05)),median=float(np.median(toff[S])),q95=float(np.quantile(toff[S],.95)),truth=float(toff[-1])),
 vacc_effort_first20days=dict(min=float(frac20[S].min()),median=float(np.median(frac20[S]))),
 u2_max=dict(q05=float(np.quantile(u2[:,S].max(0),.05)),median=float(np.median(u2[:,S].max(0))),q95=float(np.quantile(u2[:,S].max(0),.95))),
 cost_reduction=dict(q05=float(np.quantile(1-Jc[S]/J0[S],.05)),median=float(np.median(1-Jc[S]/J0[S])),q95=float(np.quantile(1-Jc[S]/J0[S],.95))),
 peakI_none_median=float(np.median(X0[:,2,S].max(0))),peakI_ctrl_median=float(np.median(X[:,2,S].max(0))),
 regret_truth_policy=dict(median=float(np.median(regret[S])),q95=float(np.quantile(regret[S],.95)),max=float(regret[S].max())))
print(json.dumps(res,indent=1)); json.dump(res,open(f'res_w{int(w1)}.json','w'),indent=1)
np.savez(f'oc_w{int(w1)}.npz',t=t,u1=u1,u2=u2,P=P,toff=toff)
