import json
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
RES = BASE / "results"


def J(f):
    txt = (RES / f).read_text()
    return json.loads(txt.replace("NaN", "null"))


def main():
    n = {}
    ex = J("figure_data_extra.json")
    n["null_in_sample"] = ex["null_all_variants"]["in_sample"]
    n["null_random"] = ex["null_all_variants"]["random_split"]
    n["null_lgo"] = ex["null_all_variants"]["leave_gene_out"]
    n["am_all_auroc"] = ex["alphamissense_all_scored"]["auroc"]
    n["am_all_n"] = ex["alphamissense_all_scored"]["n_variants"]
    n["null_random_am_variants"] = ex["alphamissense_all_scored"]["null_random_split_same_variants"]
    sh = ex["same_gene_pair_share"]
    n["pair_share_all_pct"] = 100 * sh["all_variants"]
    n["pair_share_both_pct"] = 100 * sh["genes_with_both_labels_ge3"]
    n["n_genes_all"] = sh["n_genes_all"]
    n["n_genes_both_ge3"] = sh["n_genes_evaluable"]
    n["frac_genes_pure_pct"] = 100 * sh["fraction_of_genes_label_pure"]
    n["frac_variants_pure_pct"] = 100 * sh["fraction_of_variants_in_label_pure_genes"]
    tp = ex["three_predictor_matched"]
    n["m3_pair_share_pct"] = 100 * tp["same_gene_pair_share"]
    n["m3_cross_pct"] = 100 - n["m3_pair_share_pct"]

    s = J("same_set_comparison.json")
    n["m3_n_variants"] = s["n_variants_evaluable"]
    n["m3_n_genes"] = s["n_genes_evaluable"]
    n["m3_n_intersection"] = s["n_variants_intersection"]
    for r in s["table"]:
        k = {"VARITY_R_LOO": "varity", "AlphaMissense": "am", "gMVP": "gmvp",
             "Gene-identity null (random split)": "null"}[r["predictor"]]
        n[f"m3_{k}_global"] = r["global_same_set"]
        n[f"m3_{k}_within_pooled"] = r["within_pooled"]
        n[f"m3_{k}_within_mean"] = r["within_mean"]
        if r.get("global_full_intersection") is not None:
            n[f"m3_{k}_global_full"] = r["global_full_intersection"]
    for c in s["comparisons"]:
        a, b = c["comparison"].split(" - ")
        k = {"VARITY_R_LOO": "varity", "AlphaMissense": "am", "gMVP": "gmvp",
             "gene null": "null"}
        key = f"m3_diff_{k[a]}_{k[b]}_{c['metric']}"
        n[key] = c["diff"]
        n[key + "_lo"] = c["ci_low"]
        n[key + "_hi"] = c["ci_high"]
        n[key + "_p"] = c["p"]

    d = J("dbnsfp_matched.json")
    n["p22_n_intersection"] = d["n_intersection"]
    n["p22_n_genes_intersection"] = d["n_genes_intersection"]
    n["p22_n_genes"] = d["n_genes_evaluable"]
    n["p22_n_variants"] = d["n_variants_matched"]
    n["p22_pair_share_pct"] = 100 * d["same_gene_pair_share"]
    n["p22_null_global"] = d["gene_prior_reference"]["global_matched"]
    for sch in ("submitted", "amended", "atlas", "atlas_excluding_meta",
                "submitted_high_confidence", "atlas_misfit_as_tuned",
                "amended_ensemble_named", "atlas_ensemble_named",
                "amended_ensemble_named_plus", "atlas_ensemble_named_plus",
                "amended_ensemble_flagged"):
        r = d["schemes"][sch]
        for met in ("mean", "pooled"):
            dd = r[f"did_{met}"]
            n[f"did_{sch}_{met}"] = dd["did"]
            n[f"did_{sch}_{met}_lo"] = dd["boot_both"]["ci"][0]
            n[f"did_{sch}_{met}_hi"] = dd["boot_both"]["ci"][1]
            n[f"did_{sch}_{met}_p"] = dd["boot_both"]["p"]
            n[f"did_{sch}_{met}_global_gap"] = dd["global_gap"]
            n[f"did_{sch}_{met}_within_gap"] = dd["within_gap"]
        hr = r["did_pooled_headroom"]
        n[f"did_{sch}_pooled_headroom"] = hr["did"]
        n[f"did_{sch}_pooled_headroom_p"] = hr["boot_both"]["p"]
        for met in ("submitted", "mean", "pooled"):
            n[f"rho_{sch}_{met}"] = r[met]["exposure_vs_change"]["rho"]
            n[f"rho_{sch}_{met}_p"] = r[met]["exposure_vs_change"]["p_exact"]
            n[f"rho_global_change_{sch}_{met}"] = r[met]["global_vs_change"]["rho"]
            if "exposure_vs_residual" in r[met]:
                n[f"rho_resid_{sch}_{met}"] = r[met]["exposure_vs_residual"]["rho"]
                n[f"rho_resid_{sch}_{met}_p"] = r[met]["exposure_vs_residual"]["p_exact"]
        if "did_submitted_unmatched" in r:
            n[f"did_{sch}_unmatched"] = r["did_submitted_unmatched"]
        for k2 in ("did_pooled_free_only", "did_pooled_tuned_only"):
            if k2 in r:
                n[f"{k2}_{sch}"] = r[k2]["did"]
                n[f"{k2}_{sch}_lo"] = r[k2]["boot_both"]["ci"][0]
                n[f"{k2}_{sch}_hi"] = r[k2]["boot_both"]["ci"][1]
                n[f"{k2}_{sch}_p"] = r[k2]["boot_both"]["p"]
        n[f"counts_{sch}"] = r["counts"]
    t = pd.read_csv(RES / "dbnsfp_matched.csv")
    for _, r in t.iterrows():
        k = r.predictor.replace("_score", "")
        for col in ("global_full", "global_matched", "within_mean", "within_pooled",
                    "cross_gene", "change_mean", "change_pooled", "residual_pooled"):
            n[f"p22_{k}_{col}"] = r[col]

    o = J("offset_robustness.json")
    for k, v in o["calibration"].items():
        n[f"cal_{k}"] = v
    for col, r in o["primary_test"].items():
        n[f"offset_test_{col}"] = r["clinical_minus_rest"]
        n[f"offset_test_{col}_p"] = r["p_exact"]
        n[f"offset_test_{col}_rho"] = r["atlas_level_rho"]
        n[f"offset_test_{col}_rho_p"] = r["atlas_level_p_exact"]
    for col, r in o["robustness_group_tests"].items():
        n[f"offset_test_{col}"] = r["clinical_minus_rest"]
        n[f"offset_test_{col}_p"] = r["p_exact"]
    n["offset_s4_n_genes"] = o["S4_covariates"]["n_genes"]
    for c_, v in o["S4_covariates"]["rho_path_frac_vs"].items():
        n[f"rho_pf_{c_}"] = v
    for r in o["per_predictor"]:
        k = r["predictor"].replace("_score", "")
        n[f"align_{k}"] = r["align_adj"]
        n[f"align_{k}_lo"], n[f"align_{k}_hi"] = r["align_adj_ci"]
        n[f"align_benign_{k}"] = r["align_benign"]
    for lab, r in o["S6_three_predictor_set"].items():
        if isinstance(r, dict):
            n[f"s6_{lab}"] = r["alignment"]
            n[f"s6_{lab}_lo"], n[f"s6_{lab}_hi"] = r["ci"]
            n[f"s6_{lab}_adj"] = r["alignment_covariate_adjusted"]
    s7 = o["S7_proteingym_replication"]
    n["pg_overlap_pct"] = 100 * s7["overlap_with_our_clinvar_variants"]
    n["pg_label_agree_pct"] = 100 * s7["label_agreement_on_overlap"]
    for lab in ("AlphaMissense", "VARITY_R_LOO", "gMVP"):
        n[f"s7_{lab}"] = s7[lab]["alignment"]

    ts = J("temporal_split.json")
    n["ts_new"] = ts["status_counts"]["new"]
    n["ts_same"] = ts["status_counts"]["same"]
    n["ts_changed"] = ts["status_counts"]["changed"]
    n["ts_panel_new"] = ts["panel_new_variants"]
    for t_ in ("T1_new_labels", "T2_historical_composition"):
        r = ts[t_]
        n[f"ts_{t_}_diff"] = r["clinical_minus_rest"]
        n[f"ts_{t_}_p"] = r["p_exact"]
        n[f"ts_{t_}_rho"] = r["atlas_level_rho"]
        n[f"ts_{t_}_rho_p"] = r["atlas_level_p_exact"]
        n[f"ts_{t_}_n_genes"] = r["n_genes"]
        for k, v in r["per_predictor"].items():
            n[f"ts_{t_}_{k.replace('_score', '')}"] = v
    for lab in ("VARITY_R_LOO", "gMVP", "AlphaMissense"):
        r = ts["three_predictor_new"][lab]
        n[f"ts3_{lab}"] = r["alignment"]
        n[f"ts3_{lab}_lo"], n[f"ts3_{lab}_hi"] = r["ci"]
        n["ts3_n_genes"] = r["n_genes"]
    n["ts3_n_variants"] = ts["three_predictor_new"]["n_variants"]

    c = J("gene_constraint.json")
    for k, v in c["gene_level_correlation"].items():
        for k2, v2 in v.items():
            n[f"constraint_{k}_{k2}"] = v2
    for k, v in c["variant_level_baselines"].items():
        n[f"baseline_{k}"] = v
    n["null_within_misz_decile"] = c["gene_null_within_constraint_deciles"]

    r = J("replication_matched.json")
    n["pgm_n"] = r["n_variants"]
    n["pgm_proteins"] = r["n_proteins"]
    n["pgm_overlap_pct"] = 100 * r["fraction_in_our_clinvar_table"]
    av = r["all_variants"]
    n["pgm_eval_proteins"] = av["n_proteins_evaluable"]
    n["pgm_eval_variants"] = av["n_variants_evaluable"]
    for row in av["pairwise"]:
        a, b = row["comparison"].split(" - ")
        k = {"VARITY_R_LOO": "varity", "AlphaMissense": "am", "gMVP": "gmvp"}
        n[f"pgm_{k[a]}_{k[b]}_{row['metric']}"] = row["diff"]
        n[f"pgm_{k[a]}_{k[b]}_{row['metric']}_p"] = row["p"]
    nn = r["not_in_our_table"]
    n["pgm_new_variants"] = nn["n_variants_evaluable"]
    n["pgm_new_proteins"] = nn["n_proteins_evaluable"]
    n["pgm_new_variants_all"] = nn["n_variants_all"]
    n["pgm_new_proteins_all"] = nn["n_proteins_all"]
    n["pgm_label_agreement_pct"] = 100 * r["label_agreement_on_overlap"]
    n["pgm_proteins_any_overlap_pct"] = 100 * r["fraction_of_proteins_with_any_overlap"]
    n["pgm_rows_as_submitted"] = r["n_rows_as_submitted"]
    n["pgm_dup_substitutions_as_submitted"] = r["n_substitutions_counted_more_than_once_as_submitted"]
    sub = J("independent_replication.json")
    n["pgm_submitted_evaluable_proteins"] = sub["n_proteins_evaluable"]
    mp = pd.read_parquet(BASE / "data" / "rebuild" / "pgym_clinical_mapped.parquet")
    n["pgm_mapped_variants"] = int(len(mp))
    n["pgm_mapped_proteins"] = int(mp.uniprot_acc.nunique())
    for part, tag in (("all_variants", "pgm"), ("not_in_our_table", "pgmnew")):
        for pred, v in r[part]["auroc"].items():
            for k, val in v.items():
                n[f"{tag}_auc_{pred}_{k}"] = val
    for row in nn.get("pairwise", []):
        a, b = row["comparison"].split(" - ")
        k = {"VARITY_R_LOO": "varity", "AlphaMissense": "am", "gMVP": "gmvp"}
        n[f"pgmnew_{k[a]}_{k[b]}_{row['metric']}"] = row["diff"]
        n[f"pgmnew_{k[a]}_{k[b]}_{row['metric']}_p"] = row["p"]
        n[f"pgmnew_{k[a]}_{k[b]}_{row['metric']}_lo"] = row["ci"][0]
        n[f"pgmnew_{k[a]}_{k[b]}_{row['metric']}_hi"] = row["ci"][1]
    for row in av["pairwise"]:
        a, b = row["comparison"].split(" - ")
        k = {"VARITY_R_LOO": "varity", "AlphaMissense": "am", "gMVP": "gmvp"}
        n[f"pgm_{k[a]}_{k[b]}_{row['metric']}_lo"] = row["ci"][0]
        n[f"pgm_{k[a]}_{k[b]}_{row['metric']}_hi"] = row["ci"][1]
    dm = J("dms_robustness.json")
    for unit in ("per_assay", "per_protein", "excl_top_quartile", "excl_above_median",
                 "disease_ge1", "disease_ge5", "not_disease", "function_assays",
                 "other_assays"):
        u = dm[unit]
        n[f"dms_{unit}_n"] = u["n"]
        for pair in ("AlphaMissense vs VARITY_R_LOO", "AlphaMissense vs gMVP",
                     "VARITY_R_LOO vs gMVP"):
            k = pair.replace("AlphaMissense", "am").replace("VARITY_R_LOO", "varity") \
                    .replace("gMVP", "gmvp").replace(" vs ", "_")
            n[f"dms_{unit}_{k}_p"] = u[pair]["p"]
            n[f"dms_{unit}_{k}_diff"] = u[pair]["mean_diff"]
            n[f"dms_{unit}_{k}_wins"] = f"{u[pair]['wins']}/{u[pair]['n']}"
            n[f"dms_{unit}_{k}_wins_n"] = u[pair]["wins"]
    da = J("dms_arbitration.json")
    n["dms_n_assays"] = da["n_assays_all_three"]
    n["dms_rank_agreement_global"] = da["rank_agreement_global"]
    n["dms_rank_agreement_within"] = da["rank_agreement_within_gene"]
    for p, v in da["mean_spearman"].items():
        n[f"dms_mean_rho_{p}"] = v
    n["dms_n_assays_proteingym"] = da["n_assays_proteingym"]
    n["dms_n_human"] = da["n_human_assays_total"]
    n["dms_n_mapped"] = da["n_assays_mapped"]
    n["dms_n_proteins"] = da["n_proteins"]
    n["dms_median_common"] = da["median_substitutions_scored_by_all_three"]
    n["dms_median_common_pct"] = 100 * da["median_fraction_of_assay_scored_by_all_three"]
    for p, v in da["median_coverage"].items():
        n[f"dms_median_cov_pct_{p}"] = 100 * v
    n["dms_q75_clinvar"] = dm["thresholds"]["clinvar_q75"]
    n["dms_median_clinvar"] = dm["thresholds"]["clinvar_median"]
    for k, v in dm["assay_types"].items():
        n[f"dms_assay_type_{k}"] = v
    for unit in ("per_assay", "per_protein", "excl_top_quartile", "excl_above_median",
                 "disease_ge1", "disease_ge5", "not_disease", "function_assays",
                 "other_assays"):
        for p, v in dm[unit]["mean_rho"].items():
            n[f"dms_{unit}_mean_rho_{p}"] = v
    per_prot = pd.read_csv(RES / "dms_arbitration.csv").groupby("uniprot_acc").size()
    n["dms_proteins_with_several_assays"] = int((per_prot > 1).sum())
    olda =pd.read_csv(RES / "archive_submitted" / "dms_arbitration_rows_multiplied.csv")
    n["dms_submitted_median_common"] = float(olda.n_common.median())
    n["dms_submitted_median_common_pct"] = 100 * float(olda.frac_common.median())
    old = json.loads((RES / "archive_submitted" / "dms_arbitration_rows_multiplied.json")
                     .read_text())
    for p, v in old["mean_spearman"].items():
        n[f"dms_submitted_mean_rho_{p}"] = v
    oldr = json.loads((RES / "archive_submitted" / "dms_robustness_rows_multiplied.json")
                      .read_text())
    for unit in ("per_assay", "per_protein"):
        for pair, r in oldr[unit].items():
            if isinstance(r, dict) and "p" in r:
                k = pair.replace("AlphaMissense", "am").replace("VARITY_R_LOO", "varity") \
                        .replace("gMVP", "gmvp").replace(" vs ", "_")
                n[f"dms_submitted_{unit}_{k}_p"] = r["p"]
                n[f"dms_submitted_{unit}_{k}_diff"] = r["mean_diff"]
    dsm = J("dms_robustness_snvmean.json")
    for unit in ("per_assay", "per_protein"):
        for pair, r in dsm[unit].items():
            if isinstance(r, dict) and "p" in r:
                k = pair.replace("AlphaMissense", "am").replace("VARITY_R_LOO", "varity") \
                        .replace("gMVP", "gmvp").replace(" vs ", "_")
                n[f"dms_snvmean_{unit}_{k}_p"] = r["p"]
                n[f"dms_snvmean_{unit}_{k}_diff"] = r["mean_diff"]
    n["dms_snvmean_max_abs_diff"] = max(
        abs(v[p]["mean_diff"] - dsm[u][p]["mean_diff"])
        for u, v in dm.items() if isinstance(v, dict) and "ranking" in v
        for p in ("AlphaMissense vs VARITY_R_LOO", "AlphaMissense vs gMVP",
                  "VARITY_R_LOO vs gMVP"))

    sv = J("snv_aggregation.json")
    n["snv_multi_pct"] = 100 * sv["fraction_with_more_than_one_snv"]
    n["snv_multi_n"] = sv["variants_with_more_than_one_snv"]
    pp = pd.DataFrame(sv["per_predictor"])
    n["snv_max_abs_change_diff"] = float((pp.change_pooled_mean - pp.change_pooled_first).abs().max())
    n["snv_max_abs_alignment_diff"] = float((pp.alignment_mean - pp.alignment_first).abs().max())
    n["snv_max_abs_global_diff"] = float((pp.global_mean - pp.global_first).abs().max())
    for k, v in sv["did_pooled"].items():
        n[f"snv_did_{k}_first"] = v["first"]
        n[f"snv_did_{k}_mean"] = v["mean"]
    for k, v in sv["alignment_clinical_minus_rest"].items():
        n[f"snv_align_group_{k}"] = v["difference"]
        n[f"snv_align_group_{k}_p"] = v["p_exact"]
    t3 = sv["three_predictor"]
    n["snv3_multi_pct"] = 100 * t3["fraction_with_more_than_one_snv"]
    for tag in ("first", "mean"):
        for pair in ("VARITY_minus_AlphaMissense", "VARITY_minus_gMVP"):
            for k, v in t3[tag][pair].items():
                n[f"snv3_{tag}_{pair}_{k}"] = v
        n[f"snv3_{tag}_varity_alignment"] = t3[tag]["VARITY_alignment"]

    w = J("within_gene_ranking_modern.json")
    for pair, r in w["wilcoxon"].items():
        k = pair.replace("AlphaMissense", "am").replace("VARITY_R_LOO", "varity") \
                .replace("gMVP", "gmvp").replace(" vs ", "_")
        n[f"wilcox_{k}_wins"] = r["wins"]
        n[f"wilcox_{k}_nonties"] = r["non_ties"]
        n[f"wilcox_{k}_losses"] = r["non_ties"] - r["wins"]
        n[f"wilcox_{k}_rate_pct"] = 100 * r["win_rate"]
        n[f"wilcox_{k}_loss_rate_pct"] = 100 * (1 - r["win_rate"])
        n[f"wilcox_{k}_p"] = r["p"]
    rr = J("reversal_replication.json")
    for k, v in rr["threshold_sensitivity"].items():
        n[f"threshold_{k}_genes"] = v["n_genes"]
    hh = J("head_to_head.json")
    n["loo_correction"] = hh["varity_circularity_inflation"]
    n["head_to_head_n"] = hh["n_common"]
    n["head_to_head_genes"] = hh["n_genes"]

    xc = J("extraction_check.json")
    for k, v in xc["agreement"].items():
        n[f"extract_{k.replace('_score', '')}_pearson"] = v["pearson"]
        n[f"extract_{k.replace('_score', '')}_identical_pct"] = 100 * v["identical_within_1e-3"]
        n[f"extract_{k.replace('_score', '')}_differ_pct"] = 100 * v["differ_by_more_than_0.05"]
    for k, v in xc["alignment_dbnsfp_vs_direct"].items():
        n[f"extract_align_{k.replace('_score', '')}_dbnsfp"] = v["dbnsfp"]
        n[f"extract_align_{k.replace('_score', '')}_direct"] = v["direct"]

    go = pd.read_csv(RES / "gene_constraint_offsets.csv")
    for _, r in go.iterrows():
        k = r.predictor.replace("_score", "")
        n[f"benign_vs_misz_{k}"] = r["benign_offset_vs_mis_z"]
        n[f"benign_vs_pf_{k}"] = r["benign_offset_vs_path_frac"]

    ac = pd.read_csv(RES / "dbnsfp_accounting.csv")
    n["acc_classified"] = int(len(ac))
    n["acc_in_extract"] = int(ac.in_extract.sum())
    n["acc_in_panel"] = int((ac.status == "in panel").sum())
    n["acc_lost_name_filter"] = int(ac.status.str.startswith("dropped at extraction").sum())
    n["acc_not_in_release"] = int(ac.status.str.startswith("no column").sum())
    n["acc_ensembles"] = int(ac.status.str.startswith("excluded by plan").sum())

    n["n_variants_with_constraint"] = c["n_variants_with_constraint"]
    n["s5_two_star_genes"] = o["S5_two_star"]["n_genes"]
    n["s5_two_star_variants"] = o["S5_two_star"]["n_variants"]

    n["n_variants_all"] = ex["null_all_variants"]["n_variants"]
    n["pct_genes_evaluable_3pred"] = 100 * n["m3_n_genes"] / n["n_genes_all"]
    n["pct_variants_evaluable_3pred"] = 100 * n["m3_n_variants"] / n["n_variants_all"]
    n["pct_cross_gene_pairs_both"] = 100 - n["pair_share_both_pct"]
    n["pct_cross_gene_pairs_all"] = 100 - n["pair_share_all_pct"]
    for sch in ("atlas", "atlas_excluding_meta"):
        for k2 in ("did_pooled_free_only", "did_pooled_tuned_only"):
            r = d["schemes"][sch][k2]
            n[f"{k2}_{sch}_global_gap"] = r["global_gap"]
            n[f"{k2}_{sch}_within_gap"] = r["within_gap"]
            n[f"{k2}_{sch}_n_low"] = r["n_low"]
            n[f"{k2}_{sch}_n_high"] = r["n_high"]
    full = {r["predictor"]: r["align_adj"] for r in o["per_predictor"]}
    new = ts["T1_new_labels"]["per_predictor"]
    common = sorted(set(full) & set(new))
    n["ts_full_vs_new_rho"] = float(pd.Series([full[k] for k in common]).rank().corr(
        pd.Series([new[k] for k in common]).rank()))

    mcs = J("mito_case_study.json")
    n["mito_esm2_scored"] = mcs["esm2_scored_variants"]
    n["mito_n_clusters_total"] = mcs["n_clusters"]
    for tg, sets in mcs["sets"].items():
        for s, x in sets.items():
            for k in ("n", "n_genes", "n_clusters"):
                n[f"mito_{tg}_{s}_{k}"] = x[k]
            n[f"mito_{tg}_{s}_pf_pct"] = 100 * x["pathogenic_fraction"]
            for p, v in x.items():
                if isinstance(v, dict):
                    n[f"mito_{tg}_{s}_{p}_auroc"] = v["auroc"]
                    n[f"mito_{tg}_{s}_{p}_n"] = v["n"]
    for part in ("mito_minus_control", "minus_insample_null"):
        for tg, v in mcs[part].items():
            for p, r in v.items():
                n[f"{part}_{tg}_{p}"] = r["difference"]
                n[f"{part}_{tg}_{p}_lo"] = r["ci"][0]
                n[f"{part}_{tg}_{p}_hi"] = r["ci"][1]
                n[f"{part}_{tg}_{p}_p"] = r["p"]
    n["mito_point_vs_bootstrap_mean_maxdiff"] = max(
        abs(r["difference"] - r["bootstrap_mean"])
        for part in ("mito_minus_control", "minus_insample_null")
        for v in mcs[part].values() for r in v.values())
    abl = pd.read_csv(RES / "ablation.csv")
    for _, r in abl.iterrows():
        k = r.ablation.replace(" ", "_").replace("/", "").lower()
        n[f"ablation_{k}_auroc"] = r.auroc
        n[f"ablation_{k}_delta"] = r.delta
        n[f"ablation_{k}_nfeat"] = int(r.n_features)
    sr = J("specialised_rebuild.json")
    for k, v in sr["results"].items():
        n[f"rebuild_{k}"] = v
    n["rebuild_n_proteins"] = sr["n_proteins"]
    diffs = " ".join(sr["differences_from_original"])
    n["earlier_classifier_pcs"] = int(re.search(r"(\d+) principal components", diffs).group(1))
    n["earlier_dataset_nonmito_pct"] = int(re.search(r"(\d+)% non-mitochondrial", diffs).group(1))
    n["rebuild_am_coverage_pct"] = 100 * sr["alphamissense_coverage"]
    pgm = pd.read_csv(RES / "proteingym_results.csv")
    for _, r in pgm.iterrows():
        n[f"mitodms_{r.gene}_n"] = int(r.n_substitutions)
        for m in ("AlphaMissense", "VARITY", "gMVP", "ESM-2 zero-shot"):
            n[f"mitodms_{r.gene}_{m}_rho"] = r[f"{m}_spearman"]
            n[f"mitodms_{r.gene}_{m}_n"] = int(r[f"{m}_n"])
            n[f"mitodms_{r.gene}_{m}_cov_pct"] = 100 * r[f"{m}_n"] / r.n_substitutions
    bm = json.loads((BASE / "data" / "rebuild" / "build_metadata.json").read_text())
    n["mitocarta_genes"] = bm["mitocarta_genes"]
    lab = pd.read_parquet(BASE / "data" / "rebuild" / "variants_all.parquet", columns=["label"]).label
    n["n_pathogenic_all"] = int((lab == 1).sum())
    n["n_benign_all"] = int((lab == 0).sum())
    n["ablation_old_length_auroc"] = float(re.search(
        r"AUROC (\d\.\d+)", J("ablation.json")["note"]).group(1))

    cs = J("calibration_summary.json")
    n["cal_n_variants_total"] = J("mito_case_study.json")["n_total"]
    for k, v in cs.items():
        n[f"cal_{k}_n"] = v["n"]
        n[f"cal_{k}_auroc"] = v["auroc"]
        n[f"cal_{k}_ece"] = v["ece"]
        n[f"cal_{k}_ece_scaled"] = v["ece_after_temperature"]
        n[f"cal_{k}_brier"] = v["brier"]
        n[f"cal_{k}_coverage"] = v["conformal"]["empirical_coverage"]
        n[f"cal_{k}_both_labels_pct"] = 100 * v["conformal"]["frac_both_labels"]
        pl = v["protein_level"]
        n[f"cal_{k}_proteins"] = pl["n_proteins"]
        n[f"cal_{k}_protein_median"] = pl["median_auroc"]
        n[f"cal_{k}_proteins_below_pct"] = 100 * pl["frac_proteins_below_0.7"]
    n["cal_target_coverage"] = cs["alphamissense"]["conformal"]["target_coverage"]

    ret = RES / "retained_numbers.json"
    if ret.exists():
        n.update({f"ret_{k}": v for k, v in json.loads(ret.read_text()).items()})

    out = RES / "manuscript_numbers.json"
    out.write_text(json.dumps(n, indent=1, default=float))
    print(f"{len(n)} numbers -> {out}")


if __name__ == "__main__":
    main()
