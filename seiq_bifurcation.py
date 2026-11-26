"""Bifurcation analysis of the SEIQ model (consistent version: A = mu N, DFE S = N).
Endemic equilibrium in closed form; stability from Jacobian eigenvalues."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
N=10_000; eps,alpha,mu,c=1/6,0.000015,0.00021,0.00017; A=mu*N; s,d=0.50,0.05
k1,k2,k3=eps+mu+d,alpha+mu+s,alpha+mu+c
R0f=lambda b: b*eps/(k1*k2)
def ee(b):
    R=R0f(b); S=N/R; I=mu*N*(1-1/R)/(b/R - c*(s+d*k2/eps)/k3)
    E=k2*I/eps; Q=(s*I+d*E)/k3; return np.array([S,E,I,Q])
def J(x,b):
    S,E,I,Q=x
    return np.array([[-b*I/N-mu,0,-b*S/N,c],[b*I/N,-k1,b*S/N,0],[0,eps,-k2,0],[0,d,s,-k3]])
bs=np.linspace(0.05,2.6,400); R=R0f(bs)
Iee=[];stab_ee=[];stab_dfe=[]
for b in bs:
    stab_dfe.append(np.max(np.linalg.eigvals(J([N,0,0,0],b)).real)<0)
    if R0f(b)>1:
        x=ee(b); Iee.append(x[2]); stab_ee.append(np.max(np.linalg.eigvals(J(x,b)).real)<0)
    else: Iee.append(np.nan); stab_ee.append(False)
Iee=np.array(Iee); stab_ee=np.array(stab_ee); stab_dfe=np.array(stab_dfe)
b_true=1.30
print('R0 at fitted truth',R0f(b_true),'DFE stable iff R0<1:',np.all(stab_dfe==(R<1)),
      'EE exists & stable for all R0>1:',np.all(stab_ee[R>1]), 'I* at truth',ee(b_true)[2])
# control reproduction number with constant controls (u1 vaccinates S,E; u2 treats I)
def Rc(u1,u2,b=b_true): return b*(mu/(mu+u1))*eps/((k1+u1)*(k2+u2))
u1g=np.linspace(0,0.0012,300); u2g=np.linspace(0,1.0,300); U1,U2=np.meshgrid(u1g,u2g); RC=Rc(U1,U2)
# critical constant treatment alone, vaccination alone
from scipy.optimize import brentq
u2crit=brentq(lambda u:Rc(0,u)-1,0,10); u1crit=brentq(lambda u:Rc(u,0)-1,0,1)
print('u2 crit (treatment only)',u2crit,'u1 crit (vaccination only)',u1crit)
INK,TEAL,CORAL,GREY="#14213D","#0F7173","#E76F51","#8A99A6"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":14,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":GREY,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK})
fig,ax=plt.subplots(1,2,figsize=(12.5,4.6))
m=R<1; ax[0].plot(R[m],0*R[m],color=TEAL,lw=3,label='disease-free, stable')
ax[0].plot(R[~m],0*R[~m],'--',color=GREY,lw=2.4,label='disease-free, unstable')
ax[0].plot(R[R>1],Iee[R>1],color=CORAL,lw=3,label='endemic, stable')
ax[0].axvline(1,color=INK,lw=1,ls=':'); ax[0].scatter([R0f(b_true)],[ee(b_true)[2]],s=90,color=INK,zorder=4)
ax[0].annotate('fitted model\nR₀ ≈ %.1f'%R0f(b_true),(R0f(b_true),ee(b_true)[2]),xytext=(2.5,1.0),fontsize=12,color=INK,arrowprops=dict(arrowstyle='->',color=INK))
ax[0].set_xlabel('basic reproduction number R₀'); ax[0].set_ylabel('infected at equilibrium I*'); ax[0].set_title('Forward (transcritical) bifurcation at R₀ = 1',color=INK)
ax[0].legend(frameon=False,fontsize=11,loc='upper left')
cs=ax[1].contourf(U1*1e4,U2,RC,levels=[0,0.5,1,1.5,2,2.5],colors=['#DCEFEF','#B9DFDF','#F6D5C8','#EFB39C','#E78F72'])
ax[1].contour(U1*1e4,U2,RC,levels=[1],colors=INK,linewidths=2.5)
ax[1].text(4.5,0.7,r'$R_c<1$: disease dies out',fontsize=13,color=INK); ax[1].text(0.15,0.04,r'$R_c>1$',fontsize=13,color=INK)
ax[1].set_xlabel('constant vaccination rate u₁  (×10⁻⁴ per day)'); ax[1].set_ylabel('constant treatment rate u₂'); ax[1].set_title(r'Control reproduction number $R_c(u_1,u_2)$',color=INK)
fig.colorbar(cs,ax=ax[1],label=r'$R_c$')
fig.tight_layout(); fig.savefig('fig_bifurcation.png',dpi=200)
