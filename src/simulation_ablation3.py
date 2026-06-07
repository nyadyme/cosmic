#!/usr/bin/env python3
"""
Path B (3) -- Robustness: Is fractal's short-term advantage real?
=================================================================
ablation2 showed: fractal has sharp short-term memory (100% up to LAG 5),
random a diffuse long tail. BUT: only ONE random realization.

This run tests against an ENSEMBLE:
  - random_sparse over N_RAND seeds -> mean +/- spread per LAG
  - does fractal lie outside the random band? (real effect vs chance)
  - bonus: vary n_levels -> does more depth extend the sharp horizon?
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


import time

from cosmic_web_generator import CosmicWebConfig, generate_cosmic_web, mu, eps

try:
    from sklearn.linear_model import RidgeClassifier
    HAS_SK = True
except ImportError:
    HAS_SK = False

V          = 6
LAGS       = [1, 2, 3, 5, 8, 12, 16, 20]
N_SAMPLES  = 12000
WARMUP     = 300
RHO        = 0.97
INPUT_GAIN = 0.6
RIDGE_A    = 1.0
N_RAND     = 8          # ensemble size for random_sparse
SEED       = 7


def fractal_coupling(web):
    r,c,d = [],[],[]
    for a,b,e in web.G.edges(data=True):
        r+=[a,b]; c+=[b,a]; d+=[e['W'],e['W']]
    return sp.csr_matrix((d,(r,c)), shape=(web.N,web.N))


def random_coupling(N, n_edges, pool, rng):
    r,c,d = [],[],[]; seen=set(); tries=0; pool=np.array(pool)
    while len(seen)<n_edges and tries<n_edges*20:
        i,j=int(rng.integers(0,N)),int(rng.integers(0,N)); tries+=1
        if i==j or (i,j) in seen or (j,i) in seen: continue
        seen.add((i,j)); w=float(pool[rng.integers(0,len(pool))])
        r+=[i,j]; c+=[j,i]; d+=[w,w]
    return sp.csr_matrix((d,(r,c)), shape=(N,N))


def sscale(W, rho):
    try:
        v=eigs(W.astype(float),k=1,which='LM',return_eigenvectors=False,
               tol=1e-3,maxiter=500); rho0=max(float(np.abs(v[0])),1e-6)
    except Exception:
        rho0=max(float(np.abs(W.data).max()),1e-6)
    return W*(rho/rho0)


def run_esn(W,leak,Win,inp,N):
    T=len(inp); X=np.zeros((T,N)); x=np.zeros(N)
    use = W is not None and W.nnz>0
    for t in range(T):
        u=np.zeros(V); u[inp[t]]=INPUT_GAIN
        dr=Win.dot(u)
        if use: dr=dr+W.dot(x)
        x=(1-leak)*x+leak*np.tanh(dr); X[t]=x
    return X


def racc(X,tgt):
    Xw,yw=X[WARMUP:],tgt[WARMUP:]
    Xs=(Xw-Xw.mean(0))/(Xw.std(0)+1e-8)
    n=int(0.7*len(Xs))
    if HAS_SK:
        clf=RidgeClassifier(alpha=RIDGE_A); clf.fit(_cpu(Xs[:n]),_cpu(yw[:n]))
        return clf.score(_cpu(Xs[n:]),_cpu(yw[n:]))
    Yoh=np.eye(V)[yw[:n]]
    W=np.linalg.lstsq(Xs[:n].T@Xs[:n]+RIDGE_A*np.eye(Xs.shape[1]),
                      Xs[:n].T@Yoh,rcond=None)[0]
    return float(np.mean(np.argmax(Xs[n:]@W,1)==yw[n:]))


def curve_for(W_res, leak, Win, inp, N):
    X = run_esn(W_res, leak, Win, inp, N)
    out=[]
    for lag in LAGS:
        tgt=np.zeros(N_SAMPLES,dtype=int); tgt[lag:]=inp[:-lag]
        out.append(racc(X,tgt))
    return np.array(out)


def setup(net, rng):
    web=generate_cosmic_web(net)
    N=web.N; leak=np.clip(1.0/np.asarray(web.tau),1e-3,1.0)
    mlv=int(web.levels.max())
    leaves=[c for c in range(web.n_clusters)
            if web.G.nodes[mu(c)]['level']==mlv]
    Win=np.zeros((N,V))
    for s in range(V): Win[eps(leaves[s]),s]=1.0
    Win=sp.csr_matrix(Win)
    pool=[d['W'] for _,_,d in web.G.edges(data=True)]
    return web,N,leak,Win,pool


def main():
    t0=time.time()
    rng=_np.random.default_rng(SEED)
    inp=np.asarray(rng.integers(0,V,N_SAMPLES))
    chance=1.0/V
    print(f"Robustness: V={V}, LAGs={LAGS}, random ensemble N={N_RAND}")

    # base network (n_levels=3)
    web,N,leak,Win,pool=setup(NET3:=CosmicWebConfig(
        n_levels=3,eta=4,n_top=6,tau_leaf=1.0,tau_top=40.0,
        G_min=0.1,G_iota=0.1,W_eps_iota=0.5,W_iota_mu=0.3,
        W_intra=1.0,k_neighbors=3,seed=42), rng)
    print(f"  n_levels=3: {web.n_clusters} clusters, {N} nodes, {web.n_filaments} edges")

    # fractal curve
    cf = curve_for(sscale(fractal_coupling(web),RHO), leak, Win, inp, N)

    # random ensemble
    rand_curves=[]
    for s in range(N_RAND):
        rrng=_np.random.default_rng(100+s)
        Wr=sscale(random_coupling(N,web.n_filaments,pool,rrng),RHO)
        rand_curves.append(curve_for(Wr,leak,Win,inp,N))
    rand_curves=np.array(rand_curves)
    rmean=rand_curves.mean(0); rstd=rand_curves.std(0)

    print(f"\n{'='*60}")
    print(f"FRACTAL vs RANDOM-ENSEMBLE (n_levels=3)  ({time.time()-t0:.0f}s)")
    print(f"{'='*60}")
    print(f"  {'LAG':>4}  {'fractal':>8}  {'random mean+/-std':>20}  {'verdict':>8}")
    print(f"  {'-'*50}")
    for i,lag in enumerate(LAGS):
        lo,hi = rmean[i]-rstd[i], rmean[i]+rstd[i]
        if cf[i] > hi:   verdict="FRAC+"
        elif cf[i] < lo: verdict="RAND+"
        else:            verdict="="
        print(f"  {lag:>4}  {cf[i]:>8.1%}  "
              f"{rmean[i]:>7.1%} +/- {rstd[i]:>5.1%}      {verdict:>6}")
    print(f"  {'-'*50}")
    print(f"  chance={chance:.1%} | FRAC+ = fractal above random band, RAND+ = below it")

    # depth variation: does more n_levels extend the sharp horizon?
    print(f"\n{'='*60}")
    print(f"DEPTH VARIATION: fractal curve at n_levels = 2,3,4")
    print(f"{'='*60}")
    print(f"  {'LAG':>4}  {'L=2':>8}  {'L=3':>8}  {'L=4':>8}")
    print(f"  {'-'*36}")
    depth_curves={}
    for L in [2,3,4]:
        netL=CosmicWebConfig(n_levels=L,eta=4,n_top=6,tau_leaf=1.0,tau_top=40.0,
            G_min=0.1,G_iota=0.1,W_eps_iota=0.5,W_iota_mu=0.3,
            W_intra=1.0,k_neighbors=3,seed=42)
        wL,NL,leakL,WinL,_=setup(netL,rng)
        depth_curves[L]=curve_for(sscale(fractal_coupling(wL),RHO),leakL,WinL,inp,NL)
        print(f"   (n_levels={L}: {wL.n_clusters} clusters, {NL} nodes)")
    for i,lag in enumerate(LAGS):
        print(f"  {lag:>4}  {depth_curves[2][i]:>8.1%}  "
              f"{depth_curves[3][i]:>8.1%}  {depth_curves[4][i]:>8.1%}")
    print(f"  {'-'*36}")

    # conclusion
    short = [i for i,l in enumerate(LAGS) if l<=5]
    frac_short = np.mean([cf[i] for i in short])
    rand_short = np.mean([rmean[i] for i in short])
    print(f"\n  CONCLUSION:")
    print(f"  short range (LAG<=5): fractal {frac_short:.1%} vs "
          f"random {rand_short:.1%} ({(frac_short-rand_short)*100:+.1f} PP)")
    if frac_short > rand_short + 0.03:
        print(f"  -> Fractal's short-term advantage is ROBUST against the random ensemble.")
        print(f"     The hierarchical structure contributes in the B-1-relevant regime.")
    else:
        print(f"  -> No robust advantage; was a single-case artefact.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
