"""
Aggregate per-agent, opportunity-normalised behaviour rates into group-level DEFECTIVENESS scores
and compare them with return, sustainability, equality and peace across checkpoints.

Orientation: phi is a defectiveness measure, larger = more defective.
  phi <= tau_c  -> cooperative,   phi >= tau_d  -> defective,   tau_c < tau_d

Per agent and checkpoint (episodes pooled first):
  phi_end = endangered_apples_collected / endangered_opportunities
  phi_zap = zap_successes / zap_opportunities
  phi_hybrid = w * phi_end + (1 - w) * phi_zap  (Option B: Linear Convex Combination)

usage: python aggregate_alpha.py FOLDER [--seed 3] [--T 1000] [--smooth 5] [--min-opp 10] 
                                [--zap-role defective|cooperative] [--weight-end 0.5]
                                [--win 30] [--xscale linear|log]
"""
import argparse, glob, json, re, numpy as np, pandas as pd
from scipy.stats import spearmanr

METRICS = ("return", "sustainability", "equality", "peace")
E_NUM, E_DEN, Z_NUM, Z_DEN = ("endangered_apples_collected", "endangered_opportunities", "zap_successes", "zap_opportunities")

# ---------------------------------------------------------------- loading
def load_agents(folder, seed=None):
    files = {}
    for p in glob.glob(f"{folder}/*agent_*_seed_*.json"):
        m = re.search(r"agent_(\d+)_seed_(\d+)\.json$", p)
        if m:
            files.setdefault(int(m[2]), {})[int(m[1])] = p
    if not files:
        raise SystemExit(f"no *agent_<i>_seed_<s>.json files in {folder}")
    if seed is None:
        if len(files) > 1:
            raise SystemExit(f"several seeds found {sorted(files)}; pass --seed")
        seed = next(iter(files))
    A = []
    for i in sorted(files[seed]):
        d = json.load(open(files[seed][i]))["socialjax"]
        d = next(iter(next(iter(d.values())).values()))[f"seed_{seed}"]      # env -> "ippo (agent i)" -> seed
        A.append({v["step_count"]: v for k, v in d.items() if k.startswith("step_")})
    steps = sorted(set.intersection(*[set(a) for a in A]))                   # checkpoints present for every agent
    return seed, A, steps

def counts(A, s, ep):
    miss = [k for k in (E_NUM, E_DEN, Z_NUM, Z_DEN) if k not in A[0][s]]
    if miss:
        raise SystemExit(f"agent files lack {miss}: they come from a run without the opportunity counters")
    return {k: np.array([a[s][k] for a in A], float)[:, ep] for k in (E_NUM, E_DEN, Z_NUM, Z_DEN)}   # (N, episodes)

# ------------------------- group metrics
def group_metrics(A, steps, T):
    N, out = len(A), {m: [] for m in METRICS}
    for s in steps:
        G = np.array([a[s]["return"] for a in A], float)                     # (N, episodes)
        fr = np.array([a[s]["frozen_steps"] for a in A], float)
        out["return"].append(G.mean())
        out["sustainability"].append(np.mean([a[s]["sustainability"] for a in A]))
        eq = [1 - np.abs(g[:, None] - g[None, :]).sum() / (2 * N * g.sum()) if g.sum() > 0 else 1.0 for g in G.T]
        out["equality"].append(np.mean(eq))
        out["peace"].append(np.mean(N - fr.sum(0) / (T - 1)))
    return {k: np.array(v) for k, v in out.items()}

def load_summary(folder, seed):
    ps = glob.glob(f"{folder}/*ippo_seed_{seed}.json")
    if not ps:
        return None
    d = json.load(open(ps[0]))["socialjax"]
    d = next(iter(next(iter(d.values())).values()))[f"seed_{seed}"]
    return {v["step_count"]: v for k, v in d.items() if k.startswith("step_")}

# ------------------------------------------- per-agent phi
def rate(num, den, min_opp):
    n, d = num.sum(1), den.sum(1)
    return np.where(d >= min_opp, n / np.maximum(d, 1), np.nan)

def per_agent(c, min_opp, zap_role="defective", weight_end=0.5):
    p_end = rate(c[E_NUM], c[E_DEN], min_opp)
    p_zap_raw = rate(c[Z_NUM], c[Z_DEN], min_opp)
    p_zap = p_zap_raw if zap_role == "defective" else 1 - p_zap_raw
    
    # Option B: Linear Convex Combination between endangered apple harvesting and zapping
    p_hybrid = weight_end * p_end + (1.0 - weight_end) * p_zap
    
    return {"end": p_end, "zap": p_zap, "hybrid": p_hybrid}

# ------------------------------------------------------- population aggregators
def aggregators(tc, td):
    f = {"mean": np.mean}
    return {k: (lambda x, fn=fn: fn(x[~np.isnan(x)]) if (~np.isnan(x)).any() else np.nan) for k, fn in f.items()}

# ------------------------------------------------------------------ series
def make_series(A, steps, min_opp, taus, ep, zap_role, weight_end):
    out = {}
    for s in steps:
        c = counts(A, s, ep)
        agents_phi = per_agent(c, min_opp, zap_role, weight_end)
        for kind, x in agents_phi.items():
            for name, fn in aggregators(*taus[kind]).items():
                out.setdefault(f"{kind}/{name}", []).append(fn(x))
        
        for kind in ("end", "zap"):
            num_k, den_k = (E_NUM, E_DEN) if kind == "end" else (Z_NUM, Z_DEN)
            D = c[den_k].sum()
            r = c[num_k].sum() / D if D > 0 else np.nan
            out.setdefault(f"{kind}/pooled(opp-weighted)", []).append(1 - r if (kind == "zap" and zap_role == "cooperative") else r)
            
        # Pooled hybrid score
        p_end_pooled = out["end/pooled(opp-weighted)"][-1]
        p_zap_pooled = out["zap/pooled(opp-weighted)"][-1]
        out.setdefault("hybrid/pooled(opp-weighted)", []).append(weight_end * p_end_pooled + (1.0 - weight_end) * p_zap_pooled)
        
    return {k: np.array(v) for k, v in out.items()}

def set_xaxis(ax, xscale, xmax, label="env steps"):
    from matplotlib.ticker import FuncFormatter
    ax.set_xscale(xscale)
    if xscale == "linear":
        ax.set_xlim(0, xmax)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:g}"))
        ax.set_xlabel(label + " (millions)")
    else:
        ax.set_xlabel(label)

def plot_overview(series, targets, steps, path, xscale="linear"):
    """phi aggregates on the left axis, return on the right axis."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6), sharex=True)
    
    for axi, kind in zip(ax, ("end", "zap", "hybrid")):
        names = [n for n in ("mean", "median", "min", "max") if f"{kind}/{n}" in series]
        for i, name in enumerate(names):
            axi.plot(steps, series[f"{kind}/{name}"], color=f"C{i}", lw=1.2, label=f"phi_{kind} ({name})")
        set_xaxis(axi, xscale, float(np.max(steps)))
        axi.set_ylabel(f"phi_{kind} (defectiveness)")
        axr = axi.twinx()
        axr.plot(steps, targets["return"], "k", lw=2, label="return")
        axr.set_ylabel("Mean return per agent")
        h1, l1 = axi.get_legend_handles_labels(); h2, l2 = axr.get_legend_handles_labels()
        axi.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(l1 + l2), fontsize=8, frameon=False)
        axi.set_title(f"phi_{kind} (left axis) and return (right axis)")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def rolling_corr(x, y, steps, win, detrend=False, sm_w=5):
    u, v = pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))
    if detrend:
        u = u.rolling(sm_w, center=True, min_periods=max(1, sm_w // 2)).mean().diff()
        v = v.rolling(sm_w, center=True, min_periods=max(1, sm_w // 2)).mean().diff()
    u, v, st = u.to_numpy(), v.to_numpy(), np.asarray(steps, float)
    xs, rho = [], []
    for i in range(len(u) - win + 1):
        a_, b_ = u[i:i + win], v[i:i + win]
        ok = ~np.isnan(a_) & ~np.isnan(b_)
        good = ok.sum() >= max(10, win // 2) and np.ptp(a_[ok]) > 0 and np.ptp(b_[ok]) > 0
        rho.append(spearmanr(a_[ok], b_[ok])[0] if good else np.nan)
        xs.append(st[i + win // 2])
    return np.array(xs), np.array(rho)

def plot_rolling(series, targets, steps, win, detrend, sm_w, xscale, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig, axes = plt.subplots(2, 3, figsize=(16, 6), sharex=True, sharey=True)
    for r, tn in enumerate(("return", "sustainability")):
        for c, kind in enumerate(("end", "zap", "hybrid")):
            ax = axes[r, c]
            for name in [n for n in ("mean", "median", "min", "max") if f"{kind}/{n}" in series]:
                x, rho = rolling_corr(series[f"{kind}/{name}"], targets[tn], steps, win, detrend, sm_w)
                ax.plot(x, rho, label=name, lw=1.2)
            ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-1, 1); set_xaxis(ax, xscale, float(np.max(steps))); ax.grid(alpha=0.25)
            ax.set_title(f"phi_{kind} vs {tn}")
            if r == 1: set_xaxis(ax, xscale, float(np.max(steps)), "env steps (window centre)")
            if c == 0: ax.set_ylabel("rolling Spearman rho")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle(("First-difference (trend removed)" if detrend else "Level") + f" correlation, window = {win} checkpoints")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder"); ap.add_argument("--seed", type=int)
    ap.add_argument("--T", type=int, default=1000); ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--min-opp", type=int, default=10); ap.add_argument("--zap-role", choices=("defective", "cooperative"), default="defective")
    ap.add_argument("--weight-end", type=float, default=0.5, help="Weight w for phi_end in Option B linear combination")
    ap.add_argument("--win", type=int, default=30); ap.add_argument("--xscale", choices=("linear", "log"), default="linear")
    a = ap.parse_args()
    seed, A, steps = load_agents(a.folder, a.seed)
    n_ep = len(A[0][steps[0]]["return"]); allep = np.arange(n_ep)
    print(f"seed {seed}: {len(A)} agents, {len(steps)} checkpoints, {n_ep} episodes each")

    summ = load_summary(a.folder, seed)
    if summ is not None:
        steps = [s for s in steps if s in summ]
        targets = {m: np.array([np.mean(summ[s][m]) for s in steps]) for m in METRICS}
        rebuilt = group_metrics(A, steps, a.T)
        dev = {m: float(np.max(np.abs(targets[m] - rebuilt[m]))) for m in METRICS}
        print("targets read from summary file; max |summary - rebuilt|:", {m: round(v, 5) for m, v in dev.items()})
    else:
        targets = group_metrics(A, steps, a.T)
        print("no summary file found; targets rebuilt from agent files")
    steps_arr = np.array(steps)

    pre = [per_agent(counts(A, s, allep), a.min_opp, a.zap_role, a.weight_end) for s in steps]
    taus = {k: (np.nanquantile(np.concatenate([p[k] for p in pre]), 1 / 3), np.nanquantile(np.concatenate([p[k] for p in pre]), 2 / 3))
            for k in ("end", "zap", "hybrid")}

    series = make_series(A, steps, a.min_opp, taus, allep, a.zap_role, a.weight_end)
    half1 = make_series(A, steps, a.min_opp, taus, allep[0::2], a.zap_role, a.weight_end)
    half2 = make_series(A, steps, a.min_opp, taus, allep[1::2], a.zap_role, a.weight_end)

    w, rows = a.smooth, []
    sm = lambda x: np.convolve(x, np.ones(w) / w, mode="valid")
    for sn, s in series.items():
        ok = ~np.isnan(s) & ~np.isnan(half1[sn]) & ~np.isnan(half2[sn])
        row = {"series": sn, "split_half": spearmanr(half1[sn][ok], half2[sn][ok])[0]}
        for tn, y in targets.items():
            row[f"{tn}:level"] = spearmanr(s[ok], y[ok])[0]
            row[f"{tn}:diff"] = spearmanr(np.diff(sm(s[ok])), np.diff(sm(y[ok])))[0]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("series").round(2)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(df.to_string())
    
    pd.DataFrame({"step": steps_arr, **series, **{f"T_{k}": v for k, v in targets.items()}}).to_csv(f"{a.folder}/phi_series.csv", index=False)
    df.to_csv(f"{a.folder}/alignment.csv")

    plot_overview(series, targets, steps_arr, f"{a.folder}/phi_series.pdf", a.xscale)
    if len(steps_arr) >= a.win + 5:
        plot_rolling(series, targets, steps_arr, a.win, False, a.smooth, a.xscale, f"{a.folder}/rolling_corr.pdf")
        plot_rolling(series, targets, steps_arr, a.win, True, a.smooth, a.xscale, f"{a.folder}/rolling_corr_detrended.pdf")

if __name__ == "__main__":
    main()