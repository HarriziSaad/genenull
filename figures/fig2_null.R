source("figures/theme_paper.R")

sp  <- read_json_res("split_scheme_report.json")
rob <- read_csv_res("gene_prior_robustness.csv")
cmp <- read_csv_res("compartment_generalization.csv")

st <- tibble::tibble(
  i     = 1:3,
  scheme = c("In-sample", "Random 10-fold", "Leave-gene-out"),
  auroc = c(sp$gene_prior_insample$auroc, sp$gene_prior_randomsplit$auroc,
            sp$gene_prior_leavegeneout$auroc)
)
am <- sp$alphamissense$auroc

pa <- ggplot(st) +
  annotate("rect", xmin = 0.5, xmax = 3.5, ymin = am - 0.004, ymax = am + 0.004,
           fill = PAL$gains, alpha = 0.25) +
  geom_hline(yintercept = am, colour = PAL$gains, linewidth = 1) +
  geom_hline(yintercept = 0.5, colour = PAL$muted, linewidth = 0.4,
             linetype = "dotted") +
  geom_step(aes(i - 0.5, auroc), colour = PAL$null, linewidth = 1.3,
            direction = "hv") +
  geom_rect(aes(xmin = i - 0.42, xmax = i + 0.42, ymin = 0.44, ymax = auroc),
            fill = PAL$null, alpha = 0.13) +
  geom_segment(aes(x = i - 0.42, xend = i + 0.42, y = auroc, yend = auroc),
               colour = PAL$null, linewidth = 1.6, lineend = "butt") +
  geom_text(aes(i, auroc, label = sprintf("%.3f", auroc)), vjust = -0.9,
            size = 3.1, colour = PAL$null, fontface = "bold") +
  annotate("segment", x = 2.62, xend = 2.62, y = 0.905, yend = 0.515,
           colour = PAL$loses, linewidth = 0.5,
           arrow = arrow(length = unit(5, "pt"), type = "closed")) +
  annotate("text", x = 2.55, y = 0.71, hjust = 1, size = 2.8,
           colour = PAL$loses, fontface = "bold",
           label = "held-out genes are\nunseen: the null\ncollapses to chance") +
  annotate("text", x = 3.42, y = am, hjust = 1, vjust = -0.9, size = 2.9,
           colour = PAL$gains, fontface = "bold", label = "AlphaMissense") +
  annotate("text", x = 0.6, y = 0.5, hjust = 0, vjust = -0.6, size = 2.5,
           colour = PAL$muted, label = "chance") +
  scale_x_continuous(breaks = 1:3, labels = st$scheme, expand = c(0, 0),
                     limits = c(0.5, 3.5)) +
  scale_y_continuous(limits = c(0.44, 1.0), breaks = seq(0.5, 1.0, 0.1),
                     expand = c(0, 0)) +
  labs(x = "Split scheme, strictest at right", y = "AUROC") +
  theme_paper() + grid_y_only

mg <- rob %>% mutate(margin = alphamissense_auroc - auroc)
narrow <- mg[which.min(mg$margin), ]

pb <- ggplot(mg, aes(min_variants)) +
  geom_ribbon(aes(ymin = auroc, ymax = alphamissense_auroc),
              fill = PAL$loses, alpha = 0.20) +
  geom_line(aes(y = alphamissense_auroc), colour = PAL$gains, linewidth = 1.1) +
  geom_line(aes(y = auroc), colour = PAL$null, linewidth = 1.1) +
  geom_point(aes(y = alphamissense_auroc), colour = PAL$gains, size = 2.2) +
  geom_point(aes(y = auroc), colour = PAL$null, size = 2.2) +
  geom_text(aes(y = (auroc + alphamissense_auroc) / 2,
                label = sprintf("%.3f", margin)),
            size = 2.4, colour = PAL$loses, fontface = "bold") +
  annotate("text", x = 1, y = 0.966, hjust = 0, size = 2.8,
           colour = PAL$gains, fontface = "bold", label = "AlphaMissense") +
  annotate("text", x = 1, y = 0.906, hjust = 0, size = 2.8,
           colour = PAL$null, fontface = "bold", label = "gene identity alone") +
  annotate("text", x = 50, y = 0.900, hjust = 1, size = 2.6,
           colour = PAL$loses, fontface = "bold", lineheight = 1.15,
           label = sprintf("the margin NARROWS\nas genes get better annotated:\n%.3f → %.3f",
                           mg$margin[1], narrow$margin)) +
  scale_x_log10(breaks = rob$min_variants, labels = rob$min_variants) +
  scale_y_continuous(limits = c(0.893, 0.975)) +
  labs(x = "Genes restricted to ≥ N annotated variants",
       y = "AUROC") +
  theme_paper() + grid_y_only

cm <- cmp %>%
  mutate(compartment = fct_reorder(compartment, gap),
         is_mito = compartment == "Mitochondrion",
         fill = ifelse(is_mito, PAL$loses, PAL$navy),
         a = ifelse(is_mito, 1, 0.55))

pc <- ggplot(cm, aes(gap, compartment)) +
  geom_col(aes(fill = I(fill), alpha = I(a)), width = 0.66) +
  geom_text(aes(label = sprintf("%.3f", gap)), hjust = -0.22, size = 2.6,
            colour = PAL$ink, fontface = "bold") +
  geom_vline(xintercept = median(cm$gap), colour = PAL$gains, linewidth = 0.8,
             linetype = "dashed") +
  annotate("text", x = median(cm$gap), y = 10.9, hjust = -0.08, vjust = 1,
           size = 2.6, colour = PAL$gains, fontface = "bold",
           label = sprintf("median %.3f", median(cm$gap))) +
  scale_x_continuous(limits = c(0, 0.128), expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  labs(x = "Margin of AlphaMissense over the gene-identity null", y = NULL) +
  theme_paper() + grid_x_only +
  theme(axis.text.y = element_text(
    colour = ifelse(levels(cm$compartment) == "Mitochondrion",
                    PAL$loses, PAL$muted),
    face = ifelse(levels(cm$compartment) == "Mitochondrion", "bold", "plain"),
    size = 8.5))

fig <- (pa | pb) / pc +
  plot_layout(heights = c(1, 1.15)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12, colour = PAL$navy))

save_figure(fig, "figure2_gene_identity_null", width = 7.8, height = 6.6)
