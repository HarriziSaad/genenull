source("figures/theme_paper.R")

h <- read_csv_res("head_to_head.csv")
meta <- read_json_res("head_to_head.json")

null_row <- h %>% filter(grepl("null", predictor, ignore.case = TRUE))
null_auc <- null_row$auroc[1]

is_supervised <- function(x) grepl("VARITY|gMVP", x, ignore.case = FALSE)

d <- h %>%
  mutate(is_null = grepl("null", predictor, ignore.case = TRUE),
         label   = pretty_pred(predictor),
         label   = ifelse(is_null, "gene identity alone", label),
         colour  = ifelse(is_null, PAL$null,
                          ifelse(is_supervised(predictor),
                                 PAL$loses, PAL$gains))) %>%
  arrange(auroc) %>%
  mutate(label = factor(label, levels = label))

p <- ggplot(d, aes(auroc, label)) +
  geom_segment(data = filter(d, !is_null),
               aes(x = null_auc, xend = auroc, yend = label),
               colour = PAL$grid, linewidth = 2.2, lineend = "round") +
  geom_vline(xintercept = null_auc, colour = PAL$null, linewidth = 0.9) +
  geom_linerange(aes(xmin = ci_lo, xmax = ci_hi, colour = I(colour)),
                 linewidth = 0.8) +
  geom_point(aes(colour = I(colour)), size = 3.2) +
  geom_point(size = 3.2, shape = 21, fill = NA, colour = PAL$surface,
             stroke = 0.7) +
  geom_text(data = filter(d, !is_null),
            aes(x = ci_hi, label = sprintf("  +%.3f", margin_over_null)),
            hjust = 0, size = 2.9, colour = PAL$ink, fontface = "bold") +
  annotate("text", x = null_auc, y = nrow(d) + 0.55, hjust = -0.06, vjust = 0,
           label = sprintf("%.3f", null_auc),
           size = 2.9, colour = PAL$null, fontface = "bold") +
  annotate("text", x = 0.982, y = 0.62, hjust = 1, vjust = 0,
           size = 2.6, colour = PAL$secondary,
           label = sprintf("%s variants across %s genes, one common intersection",
                           comma(meta$n_common), comma(meta$n_genes))) +
  scale_x_continuous(limits = c(0.908, 0.985), breaks = seq(0.91, 0.98, 0.02)) +
  coord_cartesian(clip = "off", ylim = c(0.35, nrow(d) + 1.1)) +
  labs(x = "AUROC", y = NULL) +
  theme_paper() + grid_x_only +
  theme(axis.text.y = element_text(colour = PAL$ink, size = 9,
                                   face = "bold"),
        plot.margin = margin(16, 14, 8, 8))

save_figure(p, "figure3_head_to_head", width = 6.4, height = 3.4)
