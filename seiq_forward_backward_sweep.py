import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
# parameters from Jonas's notebook
A,beta,eps,alpha,sigma,mu,d,c = 0.001,0.0006515,0.1667,0.000015,0.50,0.00021,0.050,0.00017
T,N=100,4000; t=np.linspace(0,T,N+1); h=t[1]-t[0]
x0=np.array([2000.,1000.,70.,20.])
A1,A2,B1,B2=1.0,1.0,2000.0,2000.0
def f(x,u1,u2):
    S,E,I,Q=x
    return np.array([A-beta*S*I-mu*S+c*Q-u1*S,
                     beta*S*I-(eps+mu+d)*E,
                     eps*E-(alpha+mu+sigma)*I-u2*I,
                     sigma*I+d*E-(alpha+mu+c)*Q+u2*I])
def g(x,l,u1,u2):  # adjoint: l' = -dH/dx
    S,E,I,Q=x; lS,lE,lI,lQ=l
    return -np.array([-(beta*I+mu+u1)*lS+beta*I*lE,
                      A1-(eps+mu+d)*lE+eps*lI+d*lQ,
                      A2-beta*S*lS+beta*S*lE-(alpha+mu+sigma+u2)*lI+(sigma+u2)*lQ,
                      c*lS-(alpha+mu+c)*lQ])
def fwd(u1,u2):
    X=np.zeros((N+1,4)); X[0]=x0
    for k in range(N):
        a=(u1[k]+u1[k+1])/2; b=(u2[k]+u2[k+1])/2
        k1=f(X[k],u1[k],u2[k]); k2=f(X[k]+h/2*k1,a,b); k3=f(X[k]+h/2*k2,a,b); k4=f(X[k]+h*k3,u1[k+1],u2[k+1])
        X[k+1]=X[k]+h/6*(k1+2*k2+2*k3+k4)
    return X
def bwd(X,u1,u2):
    L=np.zeros((N+1,4))
    for k in range(N,0,-1):
        xm=(X[k]+X[k-1])/2; a=(u1[k]+u1[k-1])/2; b=(u2[k]+u2[k-1])/2
        k1=g(X[k],L[k],u1[k],u2[k]); k2=g(xm,L[k]-h/2*k1,a,b); k3=g(xm,L[k]-h/2*k2,a,b); k4=g(X[k-1],L[k]-h*k3,u1[k-1],u2[k-1])
        L[k-1]=L[k]-h/6*(k1+2*k2+2*k3+k4)
    return L
def J(X,u1,u2):
    return np.trapezoid(A1*X[:,1]+A2*X[:,2]+B1/2*u1**2+B2/2*u2**2,t)
u1=np.zeros(N+1); u2=np.zeros(N+1); X0=fwd(u1,u2); J0=J(X0,u1,u2)
for it in range(500):
    X=fwd(u1,u2); L=bwd(X,u1,u2)
    n1=np.clip(L[:,0]*X[:,0]/B1,0,1); n2=np.clip((L[:,2]-L[:,3])*X[:,2]/B2,0,1)
    err=max(abs(n1-u1).max(),abs(n2-u2).max())
    u1=0.5*u1+0.5*n1; u2=0.5*u2+0.5*n2
    if err<1e-6: break
X=fwd(u1,u2); Jc=J(X,u1,u2)
print('iters',it,'err',err,'J0',J0,'Jc',Jc,'peakI0',X0[:,2].max(),'peakIc',X[:,2].max(),'cumI0',np.trapezoid(X0[:,2],t),'cumIc',np.trapezoid(X[:,2],t), 'u1max',u1.max(),'u2max',u2.max())
fig,ax=plt.subplots(1,3,figsize=(13,3.6))
ax[0].plot(t,X0[:,2],'--',c='gray',label='no control'); ax[0].plot(t,X[:,2],c='#1F3A5F',label='optimal control'); ax[0].set_title('Infected I(t)'); ax[0].legend()
ax[1].plot(t,X0[:,1],'--',c='gray'); ax[1].plot(t,X[:,1],c='#1F3A5F'); ax[1].set_title('Exposed E(t)')
ax[2].plot(t,u1,label='u1 vaccination'); ax[2].plot(t,u2,label='u2 treatment'); ax[2].set_title('Optimal controls'); ax[2].legend(); ax[2].set_ylim(-0.02,1.05)
for a in ax: a.set_xlabel('time (days)')
plt.tight_layout(); plt.savefig('fbs.png',dpi=150)
