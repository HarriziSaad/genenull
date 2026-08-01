source("figures/theme_paper.R")

t  <- read_csv_res("dbnsfp_common.csv")
cj <- read_json_res("dbnsfp_common.json")
did <- cj$difference_in_differences

EXP <- c("0" = "never trained on clinical labels",
         "1" = "calibration or proxy target only",
         "2" = "supervised on clinical labels")
COL <- c("0" = PAL$gains, "1" = PAL$null, "2" = PAL$loses)

t$label <- sub("_score$", "", t$predictor)
t$label <- sub("Polyphen2_HVAR", "PolyPhen-2", t$label)
t$lev   <- as.character(t$level)

pa <- ggplot(t, aes(x = change, y = fct_reorder(label, change))) +
  annotate("rect", xmin = 0, xmax = Inf, ymin = -Inf, ymax = Inf,
           fill = PAL$gains, alpha = 0.06) +
  geom_vline(xintercept = 0, colour = PAL$navy, linewidth = 0.6) +
  geom_col(aes(fill = lev), width = 0.68) +
  geom_text(aes(label = sprintf("%+.3f", change),
                hjust = ifelse(change > 0, -0.18, 1.18)),
            size = 2.4, colour = PAL$ink, fontface = "bold") +
  annotate("text", x = 0.050, y = 2.0, hjust = 1, size = 2.5,
           colour = PAL$gains, fontface = "bold",
           label = "gains under\ngene control") +
  scale_fill_manual(values = COL, labels = EXP, name = NULL) +
  scale_x_continuous(limits = c(-0.075, 0.085),
                     breaks = c(-0.05, -0.025, 0, 0.025, 0.05)) +
  labs(x = "Change in AUROC when gene identity is removed", y = NULL) +
  theme_paper() + grid_x_only +
  theme(axis.text.y = element_text(colour = PAL$ink, size = 7.5),
        legend.position = "top",
        legend.direction = "vertical",
        legend.justification = "left",
        legend.key.size = unit(9, "pt"),
        legend.text = element_text(colour = PAL$ink, size = 7.5),
        legend.margin = margin(0, 0, 4, 0))

cf <- unlist(cj$fit_coefficients)
gx <- seq(min(t$global) - 0.04, max(t$global) + 0.01, length.out = 200)
curve <- tibble::tibble(global = gx, change = cf[1] * gx^2 + cf[2] * gx + cf[3])

pb <- ggplot() +
  geom_hline(yintercept = 0, colour = PAL$navy, linewidth = 0.4,
             linetype = "dotted") +
  geom_line(data = curve, aes(global, change), colour = PAL$navy,
            linewidth = 0.7, linetype = "dashed") +
  geom_point(data = t, aes(global, change, colour = I(COL[lev])), size = 2.6) +
  geom_point(data = t, aes(global, change), shape = 21, size = 2.6,
             fill = NA, colour = PAL$surface, stroke = 0.6) +
  geom_text(data = subset(t, change > 0 | change < -0.045),
            aes(global, change, label = label, colour = I(COL[lev])),
            vjust = -1.2, size = 2.4, fontface = "bold") +
  scale_x_continuous(limits = c(0.83, 0.995)) +
  coord_cartesian(ylim = c(-0.075, 0.085)) +
  labs(x = "Global AUROC", y = "Change under gene control",
       caption = "Dashed: the change reduced accuracy alone produces, fitted on this panel.") +
  theme_paper() + grid_y_only +
  theme(plot.caption = element_text(colour = PAL$secondary, size = 7,
                                    hjust = 0, margin = margin(t = 4)),
        plot.caption.position = "plot")

dd <- tibble::tibble(
  where = factor(c("Conventional\nbenchmark", "Within genes"),
                 levels = c("Conventional\nbenchmark", "Within genes")),
  gap   = c(did$global_gap, did$within_gene_gap))

pc <- ggplot(dd, aes(where, gap)) +
  geom_hline(yintercept = 0, colour = PAL$navy, linewidth = 0.6) +
  geom_col(fill = PAL$loses, alpha = 0.85, width = 0.5) +
  geom_text(aes(label = sprintf("%.4f", gap)), vjust = -0.6,
            size = 3, colour = PAL$ink, fontface = "bold") +
  annotate("segment", x = 1.62, xend = 1.62,
           y = did$global_gap, yend = did$within_gene_gap,
           colour = PAL$gains, linewidth = 1.1,
           arrow = arrow(length = unit(5, "pt"), ends = "both")) +
  annotate("text", x = 1.70, y = mean(c(did$global_gap, did$within_gene_gap)),
           hjust = 0, size = 2.9, colour = PAL$gains, fontface = "bold",
           label = sprintf("%+.4f\n[%+.4f, %+.4f]\np < 0.0001",
                           did$difference_in_differences,
                           did$ci[1], did$ci[2])) +
  scale_y_continuous(limits = c(-0.062, 0.016)) +
  coord_cartesian(clip = "off", xlim = c(0.55, 2.9)) +
  labs(x = NULL,
       y = "Never-trained minus supervised (AUROC)",
       caption = sprintf("Bootstrapped over %s genes. Roughly half the supervised advantage is gene identity.",
                         comma(did$n_genes))) +
  theme_paper() + grid_y_only +
  theme(axis.text.x = element_text(colour = PAL$ink, size = 8, lineheight = 1.1),
        plot.caption = element_text(colour = PAL$secondary, size = 7.5,
                                    hjust = 0, margin = margin(t = 6)),
        plot.caption.position = "plot",
        plot.margin = margin(6, 74, 6, 6))

fig <- pa | (pb / pc)
fig <- fig + plot_layout(widths = c(1, 1.05)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12, colour = PAL$navy))

save_figure(fig, "figure5_dbnsfp_gradient", width = 8.4, height = 6.4)
