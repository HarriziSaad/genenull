source("figures/theme_paper.R")

cal  <- read_csv_res("calibration.csv")
summ <- read_json_res("calibration_summary.json")

PRED <- "alphamissense"
cc   <- cal %>% filter(predictor == PRED) %>% arrange(mean_score)
s    <- summ[[PRED]]

pa <- ggplot(cc, aes(mean_score, observed_rate)) +
  geom_ribbon(aes(ymin = pmin(mean_score, observed_rate),
                  ymax = pmax(mean_score, observed_rate)),
              fill = PAL$loses, alpha = 0.22) +
  geom_abline(slope = 1, intercept = 0, colour = PAL$navy,
              linewidth = 0.6, linetype = "dashed") +
  geom_line(colour = PAL$gains, linewidth = 1.2, lineend = "round") +
  geom_point(aes(size = n), colour = PAL$gains) +
  geom_point(aes(size = n), shape = 21, fill = NA, colour = PAL$surface,
             stroke = 0.7) +
  scale_size_area(max_size = 5.5, guide = "none") +
  annotate("text", x = 0.60, y = 0.20, hjust = 0, size = 2.7,
           colour = PAL$navy, label = "perfect calibration") +
  annotate("segment", x = 0.59, xend = 0.42, y = 0.205, yend = 0.40,
           colour = PAL$navy, linewidth = 0.4,
           arrow = arrow(length = unit(4, "pt"), type = "closed")) +
  annotate("text", x = 0.05, y = 0.86, hjust = 0, size = 2.8,
           colour = PAL$loses, fontface = "bold",
           label = "over-confident:\nobserved rate exceeds\npredicted probability") +
  annotate("text", x = 0.97, y = 0.06, hjust = 1, size = 2.7,
           colour = PAL$secondary,
           label = sprintf("ECE %.3f   ·   MCE %.3f   ·   Brier %.3f",
                           s$ece, s$mce, s$brier)) +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
  scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.25)) +
  coord_fixed() +
  labs(x = "Predicted pathogenicity", y = "Observed pathogenic rate") +
  theme_paper()

pb <- ggplot(cc, aes(mean_score, n)) +
  geom_col(fill = PAL$navy, alpha = 0.75, width = 0.075) +
  geom_text(aes(label = comma(n)), vjust = -0.6, size = 2.3,
            colour = PAL$secondary) +
  scale_x_continuous(limits = c(-0.03, 1.05), breaks = seq(0, 1, 0.25)) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.22)),
                     labels = label_number(scale_cut = cut_short_scale())) +
  labs(x = "Predicted pathogenicity", y = "Variants in bin") +
  theme_paper() + grid_y_only

pl <- s$protein_level
worst <- as.data.frame(pl$worst_proteins) %>% arrange(auroc) %>% head(5)

pc <- ggplot(worst, aes(auroc, fct_reorder(gene_symbol, -auroc))) +
  annotate("rect", xmin = -Inf, xmax = 0.7, ymin = -Inf, ymax = Inf,
           fill = PAL$loses, alpha = 0.06) +
  geom_vline(xintercept = 0.5, colour = PAL$muted, linewidth = 0.4,
             linetype = "dotted") +
  geom_vline(xintercept = pl$median_auroc, colour = PAL$gains,
             linewidth = 0.9) +
  geom_segment(aes(x = 0.5, xend = auroc, yend = fct_reorder(gene_symbol, -auroc)),
               colour = PAL$grid, linewidth = 1.6, lineend = "round") +
  geom_point(colour = PAL$loses, size = 3) +
  geom_text(aes(label = sprintf("%.2f  (n = %d)", auroc, n_variants)),
            hjust = -0.18, size = 2.5, colour = PAL$ink, fontface = "bold") +
  annotate("text", x = 1.16, y = 5.9, hjust = 1, vjust = 0,
           size = 2.6, colour = PAL$gains, fontface = "bold",
           label = sprintf("median protein  %.2f", pl$median_auroc)) +
  annotate("text", x = 0.06, y = 5.9, hjust = 0, vjust = 0, size = 2.6,
           colour = PAL$loses, fontface = "bold", lineheight = 1.15,
           label = sprintf("the failing tail\n%.1f%% of %s proteins < 0.70",
                           100 * pl$frac_proteins_below_0.7,
                           comma(pl$n_proteins))) +
  scale_x_continuous(limits = c(0.03, 1.16), breaks = seq(0.2, 1.0, 0.2)) +
  coord_cartesian(clip = "off", ylim = c(0.5, 7.0)) +
  labs(x = "AUROC within a single protein", y = NULL) +
  theme_paper() + grid_x_only +
  theme(axis.text.y = element_text(colour = PAL$ink, size = 9, face = "bold"))

fig <- (pa | (pb / pc + plot_layout(heights = c(0.7, 1)))) +
  plot_layout(widths = c(1, 1.05)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12, colour = PAL$navy))

save_figure(fig, "figure7_calibration", width = 7.8, height = 4.6)
