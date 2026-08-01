source("figures/theme_paper.R")

wg <- read_json_res("within_gene_ranking_all.json")
w  <- read_csv_res("within_gene_ranking_all.csv")
ir <- read_json_res("independent_replication.json")

slope <- tibble::tibble(
  predictor = pretty_pred(names(wg$global_auroc)),
  raw       = names(wg$global_auroc),
  global    = unlist(wg$global_auroc),
  within    = unlist(wg$within_gene_auroc)
) %>% mutate(change = within - global, colour = pred_colour(raw))

sl_long <- slope %>%
  pivot_longer(c(global, within), names_to = "stage", values_to = "auroc") %>%
  mutate(x = ifelse(stage == "global", 0, 1))

pa <- ggplot(sl_long, aes(x, auroc, group = predictor, colour = I(colour))) +
  geom_line(linewidth = 1.1, lineend = "round") +
  geom_point(size = 2.8) +
  geom_point(size = 2.8, shape = 21, fill = NA, colour = PAL$surface,
             stroke = 0.7) +
  geom_text(data = slope, inherit.aes = FALSE,
            aes(1, within, label = sprintf("  %s  %.3f", predictor, within),
                colour = I(colour)),
            hjust = 0, size = 2.9, fontface = "bold") +
  annotate("text", x = 0, y = 0.962, hjust = 0,
           label = "fitted to ClinVar  →  loses",
           size = 2.7, colour = PAL$loses, fontface = "bold") +
  annotate("text", x = 0, y = 0.822, hjust = 0,
           label = "never fitted to ClinVar  →  gains",
           size = 2.7, colour = PAL$gains, fontface = "bold") +
  scale_x_continuous(limits = c(-0.02, 2.35), breaks = c(0, 1),
                     labels = c("Conventional", "Gene-controlled")) +
  scale_y_continuous(limits = c(0.815, 0.968)) +
  labs(x = NULL, y = "AUROC") +
  theme_paper() + grid_y_only

hc <- read_json_res("headroom_control.json")
sw <- read_csv_res("headroom_control.csv")

obs <- bind_rows(lapply(names(hc$observed), function(n) {
  o <- hc$observed[[n]]
  tibble::tibble(predictor = pretty_pred(n), global = o$global,
                 change = o$change, expected = o$headroom_expected,
                 residual = o$residual, colour = pred_colour(n))
}))

cf <- unlist(hc$fit_coefficients)
gx <- seq(min(sw$global), max(sw$global), length.out = 200)
curve <- tibble::tibble(global = gx,
                        change = cf[1] * gx^2 + cf[2] * gx + cf[3])

pb <- ggplot() +
  geom_hline(yintercept = 0, colour = PAL$navy, linewidth = 0.4,
             linetype = "dotted") +
  geom_point(data = sw[sw$alpha > 0, ], aes(global, change),
             colour = PAL$muted, alpha = 0.35, size = 1.1) +
  geom_line(data = curve, aes(global, change), colour = PAL$navy,
            linewidth = 0.7, linetype = "dashed") +
  geom_segment(data = obs, aes(x = global, xend = global,
                               y = expected, yend = change, colour = I(colour)),
               linewidth = 0.8, alpha = 0.55) +
  geom_point(data = obs, aes(global, change, colour = I(colour)), size = 3) +
  geom_point(data = obs, aes(global, change), shape = 21, size = 3,
             fill = NA, colour = PAL$surface, stroke = 0.7) +
  geom_text(data = obs[obs$global < 0.90, ],
            aes(global, change, label = predictor, colour = I(colour)),
            vjust = -1.3, size = 2.6, fontface = "bold") +
  geom_text(data = obs[obs$global >= 0.90, ],
            aes(global, change, label = paste0(predictor, "   "),
                colour = I(colour)),
            hjust = 1, vjust = c(-0.9, 0.4, 1.5)[rank(-obs$change[obs$global >= 0.90])],
            size = 2.6, fontface = "bold") +
  scale_x_continuous(limits = c(0.5, 0.962), breaks = seq(0.5, 0.9, 0.1)) +
  coord_cartesian(ylim = c(-0.042, 0.052)) +
  labs(x = "Global AUROC", y = "Change under gene control",
       caption = "Grey: the same predictors degraded with noise.\nDashed: the change reduced accuracy alone produces.\nColoured bars: the part it cannot explain.") +
  theme_paper() + grid_y_only +
  theme(plot.caption = element_text(colour = PAL$secondary, size = 7.5,
                                    hjust = 0, margin = margin(t = 4)),
        plot.caption.position = "plot")

wx <- wg$wilcoxon
prs <- list(c("alphamissense", "varity", "AlphaMissense", "VARITY_R_LOO"),
            c("alphamissense", "gmvp",   "AlphaMissense", "gMVP"),
            c("varity", "gmvp",          "VARITY_R_LOO",  "gMVP"))
lbl <- c("AlphaMissense\nover VARITY", "AlphaMissense\nover gMVP",
         "VARITY\nover gMVP")

pc_dat <- bind_rows(lapply(seq_along(prs), function(i) {
  tibble::tibble(pair = lbl[i], diff = w[[prs[[i]][1]]] - w[[prs[[i]][2]]])
})) %>% mutate(pair = factor(pair, levels = rev(lbl)))

lookup_rate <- function(a, b) {
  fwd <- paste(a, "vs", b); rev_ <- paste(b, "vs", a)
  if (!is.null(wx[[fwd]]))  return(wx[[fwd]]$win_rate)
  if (!is.null(wx[[rev_]])) return(1 - wx[[rev_]]$win_rate)
  NA_real_
}
rate <- sapply(seq_along(prs), function(i)
  lookup_rate(prs[[i]][3], prs[[i]][4]))
ann <- tibble::tibble(pair = factor(lbl, levels = rev(lbl)),
                      lab = ifelse(is.na(rate), "",
                                   sprintf("wins %.0f%% of genes", 100 * rate)))

pc <- ggplot(pc_dat, aes(diff, pair)) +
  geom_vline(xintercept = 0, colour = PAL$navy, linewidth = 0.6) +
  geom_boxplot(outlier.shape = NA, width = 0.46, fill = "#E4EEF0",
               colour = PAL$muted, linewidth = 0.4) +
  stat_summary(fun = median, geom = "point", size = 2.2,
               colour = PAL$gains) +
  geom_text(data = ann, inherit.aes = FALSE,
            aes(x = 0.128, y = pair, label = lab),
            hjust = 1, vjust = -1.9, size = 2.6, colour = PAL$ink,
            fontface = "bold") +
  coord_cartesian(xlim = c(-0.105, 0.132)) +
  labs(x = "Per-gene difference in within-gene AUROC", y = NULL) +
  theme_paper() + grid_x_only +
  theme(axis.text.y = element_text(colour = PAL$ink, size = 8.5))

ir_slope <- tibble::tibble(
  predictor = pretty_pred(names(ir$global_auroc)),
  raw       = names(ir$global_auroc),
  global    = unlist(ir$global_auroc),
  within    = unlist(ir$within_protein_auroc)
) %>% mutate(colour = pred_colour(raw))

ir_long <- ir_slope %>%
  pivot_longer(c(global, within), names_to = "stage", values_to = "auroc") %>%
  mutate(x = ifelse(stage == "global", 0, 1))

pd <- ggplot(ir_long, aes(x, auroc, group = predictor, colour = I(colour))) +
  geom_line(linewidth = 1.1, lineend = "round") +
  geom_point(size = 2.8) +
  geom_point(size = 2.8, shape = 21, fill = NA, colour = PAL$surface,
             stroke = 0.7) +
  geom_text(data = ir_slope, inherit.aes = FALSE,
            aes(1, within, label = sprintf("  %s  %.3f", predictor, within),
                colour = I(colour)),
            hjust = 0, size = 2.9, fontface = "bold") +
  labs(caption = sprintf("ProteinGym clinical set, independently curated\n%s variants · %s proteins",
                         comma(ir$n_variants), comma(ir$n_proteins_evaluable))) +
  scale_x_continuous(limits = c(-0.02, 2.15), breaks = c(0, 1),
                     labels = c("Conventional", "Gene-controlled")) +
  scale_y_continuous(limits = c(0.9265, 0.9495)) +
  labs(x = NULL, y = "AUROC") +
  theme_paper() + grid_y_only +
  theme(plot.caption = element_text(colour = PAL$secondary, size = 7.5,
                                    hjust = 0, margin = margin(t = 4)),
        plot.caption.position = "plot")

ar <- read_json_res("dms_arbitration.json")
ac <- read_csv_res("dms_arbitration.csv")

ar_pairs <- list(c("AlphaMissense_rho", "gMVP_rho", "AlphaMissense\nover gMVP"),
                 c("VARITY_R_LOO_rho", "gMVP_rho", "VARITY\nover gMVP"),
                 c("AlphaMissense_rho", "VARITY_R_LOO_rho",
                   "AlphaMissense\nover VARITY"))
ar_lbl <- sapply(ar_pairs, function(x) x[3])
ar_dat <- bind_rows(lapply(ar_pairs, function(x)
  tibble::tibble(pair = x[3], diff = ac[[x[1]]] - ac[[x[2]]]))) %>%
  mutate(pair = factor(pair, levels = rev(ar_lbl)))

ar_p <- c(ar$pairwise$`AlphaMissense vs gMVP`$p,
          ar$pairwise$`VARITY_R_LOO vs gMVP`$p,
          ar$pairwise$`AlphaMissense vs VARITY_R_LOO`$p)
ar_ann <- tibble::tibble(
  pair = factor(ar_lbl, levels = rev(ar_lbl)),
  lab  = ifelse(ar_p < 0.05, sprintf("p = %.3f", ar_p), "not resolved"),
  col  = ifelse(ar_p < 0.05, PAL$gains, PAL$muted))

pe <- ggplot(ar_dat, aes(diff, pair)) +
  geom_vline(xintercept = 0, colour = PAL$navy, linewidth = 0.6) +
  geom_boxplot(outlier.shape = NA, width = 0.46, fill = "#E4EEF0",
               colour = PAL$muted, linewidth = 0.4) +
  stat_summary(fun = median, geom = "point", size = 2.2, colour = PAL$gains) +
  geom_text(data = ar_ann, inherit.aes = FALSE,
            aes(x = 0.20, y = pair, label = lab, colour = I(col)),
            hjust = 1, vjust = -1.9, size = 2.6, fontface = "bold") +
  coord_cartesian(xlim = c(-0.17, 0.205)) +
  labs(x = "Per-assay difference in agreement with experiment (Spearman)",
       y = NULL,
       caption = sprintf("%s human DMS assays. The two comparisons experiment resolves both favour\nthe gene-controlled ranking; conventional AUROC ranks gMVP first, experiment ranks it last.",
                         ar$n_assays_all_three)) +
  theme_paper() + grid_x_only +
  theme(axis.text.y = element_text(colour = PAL$ink, size = 8.5),
        plot.caption = element_text(colour = PAL$secondary, size = 7.5,
                                    hjust = 0, margin = margin(t = 4)),
        plot.caption.position = "plot")

fig <- (pa | pb) / (pc | pd) / pe +
  plot_layout(heights = c(1.05, 0.95, 0.62)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12, colour = PAL$navy))

save_figure(fig, "figure4_ranking_reversal", width = 7.6, height = 8.6)
