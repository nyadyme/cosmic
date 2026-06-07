#!/usr/bin/env python3
"""
Resolution sweep: How do capabilities scale with fractal resolution?
=====================================================================
Varies n_levels (depth) and eta (branching) and measures per configuration:
  - Memory Capacity (Delayed-Copy, ESN reservoir)  -> capability
  - tau spectrum (number/span of timescales)
  - hardware: total wire length, REDAC-6 tile cut
  - fractal vs matched random-sparse: does the gap change with scale?

Answers honestly (against B-3/B-4): does more resolution bring capability, and
is the gain fractal-specific or generic size scaling?
"""

try:
    import cupy as np
    import cupyx.scipy.sparse as sp
    from cupyx.scipy.sparse.linalg import eigs
    _GPU = True
except ImportError:
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigs
    _GPU = False

import numpy as _np


def _cpu(a):
    if _GPU and hasattr(a, 'get'):
        return a.get()
    return _np.asarray(a)


from sklearn.cluster import KMeans
import time

from cosmic_web_generator import CosmicWebConfig, generate_cosmic_web, mu, eps

V          = 6
LAGS       = [1, 2, 3, 5, 8, 12, 16, 20]
N_SAMPLES  = 8000
WARMUP     = 250
RHO        = 0.97
INPUT_GAIN = 0.6
RIDGE_A    = 1.0
chance     = 1.0 / V

from sklearn.linear_model import RidgeClassifier


def base_cfg(n_levels, eta):
    return CosmicWebConfig(
        n_levels=n_levels, eta=eta, n_top=6,
        tau_leaf=1.0, tau_top=40.0,
        G_min=0.1, G_iota=0.1, W_eps_iota=0.5, W_iota_mu=0.3,
        W_intra=1.0, k_neighbors=3, seed=42)


def fractal_W(web):
    r,c,d = [],[],[]
    for a,b,e in web.G.edges(data=True):
        r+=[a,b]; c+=[b,a]; d+=[e['W'],e['W']]
    return sp.csr_matrix((d,(r,c)), shape=(web.N,web.N))


def random_W(N, n_edges, pool, rng):
    seen=set(); tries=0; pool=np.array(pool); r,c,d=[],[],[]
    while len(seen)<n_edges and tries<n_edges*20:
        i,j=int(rng.integers(0,N)),int(rng.integers(0,N)); tries+=1
        if i!=j and (i,j) not in seen and (j,i) not in seen:
            seen.add((i,j)); w=float(pool[rng.integers(0,len(pool))])
            r+=[i,j]; c+=[j,i]; d+=[w,w]
    return sp.csr_matrix((d,(r,c)), shape=(N,N)), list(seen)


def sscale(W, rho):
    try:
        v=eigs(W.astype(float),k=1,which='LM',return_eigenvectors=False,
               tol=1e-3,maxiter=500); rho0=max(float(np.abs(v[0])),1e-6)
    except Exception:
        rho0=max(float(np.abs(W.data).max()),1e-6)
    return W*(rho/rho0)


def esn(W,leak,Win,inp,N):
    X=np.zeros((len(inp),N)); x=np.zeros(N)
    use=W is not None and W.nnz>0
    for t in range(len(inp)):
        u=np.zeros(V); u[inp[t]]=INPUT_GAIN
        dr=Win.dot(u)
        if use: dr=dr+W.dot(x)
        x=(1-leak)*x+leak*np.tanh(dr); X[t]=x
    return X


def racc(X,tgt):
    Xw,yw=X[WARMUP:],tgt[WARMUP:]
    Xs=(Xw-Xw.mean(0))/(Xw.std(0)+1e-8); n=int(0.7*len(Xs))
    clf=RidgeClassifier(alpha=RIDGE_A); clf.fit(_cpu(Xs[:n]),_cpu(yw[:n]))
    return clf.score(_cpu(Xs[n:]),_cpu(yw[n:]))


def memory_capacity(W, leak, Win, inp, N):
    X=esn(W,leak,Win,inp,N)
    mc=0.0
    for lag in LAGS:
        tgt=np.zeros(len(inp),dtype=int); tgt[lag:]=inp[:-lag]
        mc += max(racc(X,tgt)-chance,0)
    return mc/(1-chance)


def hardware(web, rng):
    pos=np.asarray(web.positions); N=web.N
    fe=[(a,b) for a,b,_ in web.G.edges(data=True)]
    fl=np.linalg.norm(pos[[e[0] for e in fe]]-pos[[e[1] for e in fe]],axis=1)
    lab=KMeans(n_clusters=6,n_init=3,random_state=0).fit_predict(_cpu(pos))
    cut=sum(1 for i,j in fe if lab[i]!=lab[j])/len(fe)
    # random_global wire
    _,re=random_W(N,len(fe),[1],rng)
    rl=np.linalg.norm(pos[[e[0] for e in re]]-pos[[e[1] for e in re]],axis=1)
    return fl.sum(), cut, rl.sum()/max(fl.sum(),1e-9)


def run_config(n_levels, eta, inp):
    web=generate_cosmic_web(base_cfg(n_levels,eta))
    N=web.N
    leak=np.clip(1.0/np.asarray(web.tau),1e-3,1.0)
    n_tau=len(np.unique(np.round(web.tau,3)))
    mlv=int(web.levels.max())
    leaves=[c for c in range(web.n_clusters)
            if web.G.nodes[mu(c)]['level']==mlv]
    Win=np.zeros((N,V))
    for s in range(min(V,len(leaves))): Win[eps(leaves[s]),s]=1.0
    Win=sp.csr_matrix(Win)
    pool=[d['W'] for _,_,d in web.G.edges(data=True)]
    rng=_np.random.default_rng(11)

    mc_f = memory_capacity(sscale(fractal_W(web),RHO), leak, Win, inp, N)
    Wr,_ = random_W(N, web.n_filaments, pool, rng)
    mc_r = memory_capacity(sscale(Wr,RHO), leak, Win, inp, N)
    wl, cut, wfac = hardware(web, _np.random.default_rng(22))

    return dict(L=n_levels, eta=eta, ncl=web.n_clusters, N=N,
                n_edges=web.n_filaments, n_tau=n_tau,
                tau_max=web.tau.max(), mc_f=mc_f, mc_r=mc_r,
                wire=wl, cut=cut, wfac=wfac)


def main():
    t0=time.time()
    rng=_np.random.default_rng(7)
    inp=np.asarray(rng.integers(0,V,N_SAMPLES))
    print(f"Resolution sweep: V={V}, {N_SAMPLES} samples, MC across LAGs={LAGS}")

    configs = [(2,4),(3,4),(4,4),   # depth sweep
               (3,3),(3,5)]          # branching sweep (L=3,eta=4 included above)
    rows=[]
    for L,eta in configs:
        print(f"  config n_levels={L}, eta={eta} ...")
        rows.append(run_config(L,eta,inp))

    print(f"\n{'='*78}")
    print(f"RESOLUTION SWEEP  ({time.time()-t0:.0f}s)")
    print(f"{'='*78}")
    print(f"  {'L':>2} {'eta':>3} {'clusters':>7} {'nodes':>7} {'#tau':>5} "
          f"{'MC_frac':>8} {'MC_rand':>8} {'wire':>8} {'Tile%':>6} {'wfac':>5}")
    print(f"  {'-'*72}")
    for r in rows:
        print(f"  {r['L']:>2} {r['eta']:>3} {r['ncl']:>7} {r['N']:>7} "
              f"{r['n_tau']:>5} {r['mc_f']:>8.2f} {r['mc_r']:>8.2f} "
              f"{r['wire']:>8.0f} {r['cut']:>6.1%} {r['wfac']:>5.1f}")
    print(f"  {'-'*72}")
    print(f"  MC = Memory Capacity (higher=more memory) | "
          f"Tile%=inter-tile cut | wfac=random/fractal wire")

    # analysis
    print(f"\n  ANALYSIS:")
    depth = [r for r in rows if r['eta']==4]
    depth.sort(key=lambda r:r['L'])
    print(f"  depth sweep (eta=4): MC_frac " +
          " -> ".join(f"L{r['L']}={r['mc_f']:.2f}" for r in depth))
    print(f"    node count:         " +
          " -> ".join(f"L{r['L']}={r['N']}" for r in depth))
    # fractal vs random gap
    gaps = [(r['mc_f']-r['mc_r']) for r in rows]
    print(f"  fractal-random MC gap: min={min(gaps):+.2f}, "
          f"max={max(gaps):+.2f}, mean={np.mean(gaps):+.2f}")
    if np.mean(gaps) < 0.2 and np.mean(gaps) > -0.2:
        print(f"    -> gap stays ~0 across all resolutions:")
        print(f"       More resolution = generic scaling, NOT fractal-specific.")
    # hardware scaling
    print(f"  Hardware: wire advantage (wfac) at largest network "
          f"N={max(r['N'] for r in rows)}: "
          f"{max(rows,key=lambda r:r['N'])['wfac']:.1f}x")
    big = max(rows,key=lambda r:r['N'])
    print(f"    tile cut at N={big['N']}: {big['cut']:.1%} "
          f"(stays low = locality scales along)")
    print(f"{'='*78}")


if __name__ == '__main__':
    main()
