import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[2]
RES = BASE / "results" / "rebuild"

MIN_PER_CLASS = 3
RNG = np.random.default_rng(42)
N_REPEATS = 5
LEVELS = np.round(np.arange(0.0, 0.96, 0.05), 2)

EXPOSURE = {
    'SIFT':           (0, 'none; conservation only'),
    'PolyPhen-2':     (0, 'none w.r.t. modern ClinVar'),
    'ESM-2 zero-shot': (0, 'none; no clinical labels'),
    'AlphaMissense':  (1, 'calibrated on, not fitted to'),
    'VARITY_R_LOO':   (2, 'supervised on ClinVar'),
    'gMVP':           (2, 'supervised on ClinVar'),
}

def log(m):
    print(f"[{datetime.now():%H:%M:%S}] {m}", flush=True)

def within_gene_mean(d, col, _cache={}):
    key = id(d)
    if key not in _cache:
        g = d.gene_symbol.values
        y = d.label.values.astype(np.int64)
        n_pos = pd.Series(y).groupby(g).transform('sum').values
        n = pd.Series(y).groupby(g).transform('size').values
        keep = (n_pos >= MIN_PER_CLASS) & ((n - n_pos) >= MIN_PER_CLASS)
        _cache[key] = (g[keep], y[keep], keep)
    gk, yk, keep = _cache[key]

    s = pd.Series(d[col].values[keep])
    rank = s.groupby(gk).rank(method='average').values
    pos_rank = pd.Series(np.where(yk == 1, rank, 0.0)).groupby(gk).sum()
    npos = pd.Series(yk).groupby(gk).sum()
    ntot = pd.Series(yk).groupby(gk).size()
    nneg = ntot - npos
    auc = (pos_rank - npos * (npos + 1) / 2) / (npos * nneg)
    return float(auc.mean()), int(len(auc))

def degrade(scores, alpha, rng):
    r = pd.Series(scores).rank(pct=True).values
    return (1 - alpha) * r + alpha * rng.random(len(r))

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    mod = __import__('15_within_gene_ranking')

    d, cols = mod.load()
    d = d.dropna(subset=cols).reset_index(drop=True)
    pretty = dict(mod.PREDICTORS)
    log(f"five-predictor intersection: {len(d):,} variants, "
        f"{d.gene_symbol.nunique():,} genes")

    c0 = cols[0]
    slow = []
    for _, grp in d.groupby('gene_symbol', sort=False):
        yy = grp.label.values
        if yy.sum() < MIN_PER_CLASS or len(yy) - yy.sum() < MIN_PER_CLASS:
            continue
        slow.append(roc_auc_score(yy, grp[c0].values))
    fast, n_fast = within_gene_mean(d, c0)
    assert n_fast == len(slow), (n_fast, len(slow))
    assert abs(fast - float(np.mean(slow))) < 1e-9, (fast, np.mean(slow))
    log(f"vectorised within-gene AUROC agrees with roc_auc_score "
        f"({fast:.6f}, {n_fast:,} genes)")

    observed = {}
    for c in cols:
        g = float(roc_auc_score(d.label, d[c]))
        w, n_eval = within_gene_mean(d, c)
        observed[pretty[c]] = {'global': g, 'within': w, 'change': w - g}
        log(f"  observed  {pretty[c]:<16} global {g:.4f}  within {w:.4f}  "
            f"change {w - g:+.4f}")
    log(f"  ({n_eval:,} evaluable genes)")

    log("\ndegradation sweep (noise injected on the rank scale)")
    rows = []
    for c in cols:
        for alpha in LEVELS:
            gs, ws = [], []
            for rep in range(1 if alpha == 0 else N_REPEATS):
                rng = np.random.default_rng(1000 + int(alpha * 100) * 17 + rep)
                d['_deg'] = degrade(d[c].values, alpha, rng)
                gs.append(float(roc_auc_score(d.label, d['_deg'])))
                ws.append(within_gene_mean(d, '_deg')[0])
            rows.append({'predictor': pretty[c], 'alpha': float(alpha),
                         'global': float(np.mean(gs)),
                         'within': float(np.mean(ws)),
                         'change': float(np.mean(ws) - np.mean(gs))})
        log(f"  {pretty[c]:<16} swept {len(LEVELS)} levels")
    sweep = pd.DataFrame(rows)
    d.drop(columns='_deg', inplace=True)

    fit = sweep[sweep.alpha > 0]
    coef = np.polyfit(fit['global'], fit['change'], 2)
    predict = np.poly1d(coef)
    log(f"\nheadroom null fitted on {len(fit)} degraded points "
        f"(quadratic in global AUROC)")

    resid = {}
    for name, o in observed.items():
        exp_change = float(predict(o['global']))
        resid[name] = {**o, 'headroom_expected': exp_change,
                       'residual': o['change'] - exp_change,
                       'exposure': EXPOSURE.get(name, (None, 'unknown'))[0],
                       'exposure_label': EXPOSURE.get(name, (None, 'unknown'))[1]}

    log("\nOBSERVED vs HEADROOM-PREDICTED CHANGE")
    log(f"  {'predictor':<16} {'global':>7} {'change':>8} {'headroom':>9} "
        f"{'residual':>9}  exposure")
    for name in sorted(resid, key=lambda k: -resid[k]['residual']):
        r = resid[name]
        log(f"  {name:<16} {r['global']:>7.4f} {r['change']:>+8.4f} "
            f"{r['headroom_expected']:>+9.4f} {r['residual']:>+9.4f}  "
            f"{r['exposure_label']}")

    from scipy.stats import spearmanr
    names = [n for n in resid if resid[n]['exposure'] is not None]
    expo = [resid[n]['exposure'] for n in names]
    raw = [resid[n]['change'] for n in names]
    res = [resid[n]['residual'] for n in names]
    glob = [resid[n]['global'] for n in names]

    rho_raw = spearmanr(expo, raw)
    rho_res = spearmanr(expo, res)
    rho_conf = spearmanr(glob, raw)
    log("\nSPEARMAN CORRELATIONS")
    log(f"  exposure vs raw change        rho = {rho_raw.statistic:+.3f} "
        f"p = {rho_raw.pvalue:.4f}   <- the manuscript's claim")
    log(f"  global AUROC vs raw change    rho = {rho_conf.statistic:+.3f} "
        f"p = {rho_conf.pvalue:.4f}   <- the confound")
    log(f"  exposure vs residual          rho = {rho_res.statistic:+.3f} "
        f"p = {rho_res.pvalue:.4f}")
    log("  NOTE: the rank correlation is preserved trivially, because the "
        "headroom")
    log("  correction is monotone in a quantity that is itself perfectly "
        "rank-correlated")
    log("  with the outcome. The rank test is therefore NOT the evidence. "
        "Magnitude is:")

    span_obs = max(raw) - min(raw)
    span_head = float(predict(max(glob)) - predict(min(glob)))
    log(f"\n  observed span of change across predictors   {span_obs:+.4f}")
    log(f"  span headroom alone can produce             {span_head:+.4f}")
    log(f"  ratio                                       {abs(span_obs / span_head):.1f}x")
    log(f"  headroom explains {abs(100 * span_head / span_obs):.1f}% of the spread")

    log("\n  gene-level bootstrap on the residuals (2,000 replicates)")
    gk, yk, keep = None, None, None
    genes = d.gene_symbol.values
    uniq = pd.unique(genes)
    gidx = {g: np.where(genes == g)[0] for g in uniq}
    boot_rng = np.random.default_rng(7)
    resid_boot = {pretty[c]: [] for c in cols}
    for _ in range(2000):
        pick = boot_rng.choice(uniq, len(uniq), replace=True)
        rows_ = np.concatenate([gidx[g] for g in pick])
        sub = d.iloc[rows_].copy()
        sub['gene_symbol'] = np.concatenate(
            [np.full(len(gidx[g]), f"{g}__{i}") for i, g in enumerate(pick)])
        for c in cols:
            gA = float(roc_auc_score(sub.label, sub[c]))
            wA, _ = within_gene_mean(sub, c)
            resid_boot[pretty[c]].append((wA - gA) - float(predict(gA)))
        within_gene_mean.__defaults__[0].clear()
    for name in resid:
        v = np.array(resid_boot[name])
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        resid[name]['residual_ci'] = [lo, hi]
        resid[name]['residual_excludes_zero'] = bool(lo > 0 or hi < 0)
        log(f"    {name:<16} residual {resid[name]['residual']:+.4f} "
            f"[{lo:+.4f}, {hi:+.4f}]"
            f"{'  *' if (lo > 0 or hi < 0) else ''}")

    log("\nMATCHED PAIRS (similar global AUROC, different ClinVar exposure)")
    pairs = []
    ns = list(observed)
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            a, b = ns[i], ns[j]
            if EXPOSURE.get(a, (None,))[0] == EXPOSURE.get(b, (None,))[0]:
                continue
            gap = abs(observed[a]['global'] - observed[b]['global'])
            if gap > 0.02:
                continue
            rec = {'a': a, 'b': b, 'global_gap': gap,
                   'change_a': observed[a]['change'],
                   'change_b': observed[b]['change'],
                   'change_gap': observed[a]['change'] - observed[b]['change']}
            pairs.append(rec)
            log(f"  {a} ({observed[a]['global']:.4f}, {observed[a]['change']:+.4f}) "
                f"vs {b} ({observed[b]['global']:.4f}, {observed[b]['change']:+.4f})")
            log(f"     global differs by {gap:.4f}, change differs by "
                f"{rec['change_gap']:+.4f}")

    sweep.to_csv(RES / 'headroom_control.csv', index=False)
    out = {
        'n_variants': int(len(d)),
        'n_genes_evaluable': int(n_eval),
        'levels': [float(x) for x in LEVELS],
        'n_repeats': N_REPEATS,
        'fit_coefficients': [float(x) for x in coef],
        'fit_n_points': int(len(fit)),
        'observed': resid,
        'spearman': {
            'exposure_vs_raw_change': [float(rho_raw.statistic), float(rho_raw.pvalue)],
            'global_vs_raw_change': [float(rho_conf.statistic), float(rho_conf.pvalue)],
            'exposure_vs_residual': [float(rho_res.statistic), float(rho_res.pvalue)],
            'rank_test_caveat':
                'The rank correlation with the residual is preserved trivially, '
                'because the correction is monotone in global AUROC and global '
                'AUROC is perfectly rank-correlated with the raw change. The rank '
                'test is not the evidence; the magnitude comparison below is.',
        },
        'magnitude': {
            'observed_span': float(span_obs),
            'headroom_span': float(span_head),
            'ratio': float(abs(span_obs / span_head)),
            'percent_of_spread_explained_by_headroom':
                float(abs(100 * span_head / span_obs)),
        },
        'matched_pairs': pairs,
        'note': 'Noise is injected on the rank scale over the whole score column, '
                'degrading between-gene and within-gene discrimination together, '
                'so the sweep traces what a genuinely weaker predictor does.',
        'caveat': 'Six predictors is a small basis for a rank correlation. The '
                  'matched-pair comparison is model-free and does not depend on '
                  'the fitted curve.',
    }
    (RES / 'headroom_control.json').write_text(json.dumps(out, indent=2))
    log(f"\nwrote {RES / 'headroom_control.csv'}")
    log(f"wrote {RES / 'headroom_control.json'}")

if __name__ == '__main__':
    main()
