"""Generate the README banner (assets/readme/banner.png).  Usage: python scripts/make_banner.py"""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import optics
OUT = Path(__file__).resolve().parents[1] / "assets" / "readme"
OUT.mkdir(parents=True, exist_ok=True)
NAVY="#14213d"; BLUE="#4c8dff"; ORANGE="#f08a24"; GREY="#8d99ae"
# beam envelope (tuned) along the line
def env(k):
    L=optics.LAYOUT; b,a,e=optics.BEAM["beta"],optics.BEAM["alpha"],optics.BEAM["emit"]
    S=[e*np.array([[b,-a],[-a,(1+a*a)/b]]) for _ in range(2)]
    segs=[("d",L["d0"]),("q",0),("d",L["d1"]),("q",1),("d",L["d2"]),("q",2),("d",L["d3"])]
    s=0; out=[(0,np.sqrt(S[0][0,0]),np.sqrt(S[1][0,0]))]
    for kind,v in segs:
        ln=v if kind=="d" else L["lq"]; n=60
        for _ in range(n):
            dl=ln/n
            for p in (0,1):
                M=optics.drift(dl) if kind=="d" else optics.quad(dl,k[v])[p]
                S[p]=M@S[p]@M.T
            s+=dl; out.append((s,np.sqrt(S[0][0,0]),np.sqrt(S[1][0,0])))
    return np.array(out)
E=env((4.9,-13.5,11.9))
fig=plt.figure(figsize=(16,4.2),dpi=100); fig.patch.set_facecolor(NAVY)
ax=fig.add_axes([0,0,1,1]); ax.set_facecolor(NAVY); ax.set_xlim(0,16); ax.set_ylim(0,4.2); ax.axis("off")
# beam on the right half: map s in [0,3.04] -> x in [8.2,15.6]; envelope as filled band
x=8.2+E[:,0]/E[-1,0]*7.4
for col,sig,off in [(BLUE,E[:,1],2.55),(ORANGE,E[:,2],1.25)]:
    h=sig/E[:,1:].max()*0.75
    for w,alpha in [(1.0,0.10),(0.75,0.18),(0.5,0.35)]:
        ax.fill_between(x,off-h*w,off+h*w,color=col,alpha=alpha,lw=0)
    ax.plot(x,off+h,color=col,lw=1.6); ax.plot(x,off-h,color=col,lw=1.6)
for q0 in (0.5,1.02,1.54):
    xq=8.2+q0/E[-1,0]*7.4; wq=0.12/E[-1,0]*7.4
    ax.add_patch(Rectangle((xq,0.25),wq,3.6,color=GREY,alpha=0.25,lw=0))
ax.add_patch(Rectangle((15.55,0.25),0.06,3.6,color="white",alpha=0.8,lw=0))
ax.text(8.25,3.75,"σx",color=BLUE,fontsize=13,weight="bold"); ax.text(8.25,0.35,"σy",color=ORANGE,fontsize=13,weight="bold")
# faint NN in background left
rng=np.random.default_rng(0)
layers=[[(6.55,y) for y in (1.4,2.1,2.8)],[(7.1,y) for y in np.linspace(0.9,3.3,5)],[(7.65,y) for y in (1.75,2.45)]]
for A,B in zip(layers[:-1],layers[1:]):
    for p in A:
        for q in B: ax.plot([p[0],q[0]],[p[1],q[1]],color="white",alpha=0.12,lw=0.8)
for L_ in layers:
    for p in L_: ax.add_patch(Circle(p,0.09,color="white",alpha=0.35))
ax.text(0.6,2.75,"Machine Learning for",color="white",fontsize=30,weight="bold",va="center")
ax.text(0.6,2.05,"Accelerator Physics",color="white",fontsize=30,weight="bold",va="center")
ax.plot([0.62,2.4],[1.55,1.55],color=ORANGE,lw=4)
ax.text(0.6,1.1,"Beginner lecture · hands-on Colab notebooks with Cheetah",color="#cbd5e1",fontsize=15,va="center")
ax.text(0.6,0.6,"H. Guler · IJCLab (CNRS/IN2P3, Université Paris-Saclay)",color="#94a3b8",fontsize=12.5,va="center")
fig.savefig(OUT / "banner.png", dpi=100, facecolor=NAVY)
print("wrote", OUT / "banner.png")
