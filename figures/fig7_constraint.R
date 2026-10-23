source("figures/theme_final.R")

g <- read_csv_res("gene_constraint_genes.csv") %>% filter(n >= 20, !is.na(mis_z))
j <- read_json_res("gene_constraint.json")
gl <- j$gene_level_correlation[["genes_n>=20"]]
stopifnot(nrow(g) == gl$n_genes)

XL <- c(-4, 9)
inr <- g %>% filter(mis_z >= XL[1], mis_z < XL[2])
trend <- inr %>%
  mutate(bin = cut(mis_z, quantile(mis_z, seq(0, 1, 0.1)), include.lowest = TRUE)) %>%
  group_by(bin) %>%
  summarise(x = median(mis_z), m = mean(path_frac), lo = quantile(path_frac, 0.25),
            hi = quantile(path_frac, 0.75), .groups = "drop")
BX <- 0.5; BY <- 0.05
tiles <- inr %>%
  mutate(bx = XL[1] + (floor((mis_z - XL[1]) / BX) + 0.5) * BX,
         by = pmin(floor(path_frac / BY), 1 / BY - 1) * BY + BY / 2) %>%
  count(bx, by) %>% mutate(id = row_number())
stopifnot(sum(tiles$n) == nrow(inr))
cells <- tiles %>% rowwise() %>%
  do(tibble(id = .$id, n = .$n,
            x = .$bx + c(-1, 1, 1, -1) * BX * 0.44,
            y = .$by + c(-1, -1, 1, 1) * BY * 0.42)) %>% ungroup()
pa <- ggplot() +
  ggforce::geom_shape(data = cells, aes(x, y, group = id, fill = n),
                      radius = unit(0.9, "pt")) +
  geom_segment(data = trend, aes(x = x, xend = x, y = lo, yend = hi), colour = INK,
               linewidth = 0.45, alpha = 0.75) +
  geom_line(data = trend, aes(x, m), colour = INK, linewidth = 0.8) +
  geom_point(data = trend, aes(x, m), shape = 21, fill = INK, colour = SURF, stroke = 0.6,
             size = 2.0) +
  annotate("text", x = XL[2] - 0.15, y = 1.035, hjust = 1, vjust = 0, size = 2.2, colour = INK,
           label = sprintf("Spearman \u03c1 = %.2f over %s genes", gl$path_frac_vs_mis_z,
                           comma(gl$n_genes))) +
  scale_fill_gradientn(colours = JADE_RAMP, trans = "log10", name = "Genes per cell",
                       breaks = c(1, 3, 10, 30),
                       guide = guide_colourbar(barheight = unit(4, "pt"),
                                               barwidth = unit(78, "pt"),
                                               title.position = "left", title.vjust = 1.1,
                                               ticks.colour = SURF)) +
  scale_y_continuous(labels = percent, breaks = seq(0, 1, 0.25), expand = c(0.01, 0)) +
  scale_x_continuous(breaks = seq(-4, 8, 2), labels = function(x) minus(as.character(x)),
                     expand = c(0, 0)) +
  coord_cartesian(xlim = XL, ylim = c(0, 1), clip = "off") +
  labs(x = "gnomAD v4.1 missense Z (more constrained \u2192)",
       y = "Pathogenic fraction of the gene's variants") +
  theme_final() + no_grid + top_left_legend + theme(plot.margin = margin(6, 10, 6, 6))

vb <- j$variant_level_baselines
rb <- tibble(
  lab = c("missense Z", "missense o/e, inverted", "LOEUF, inverted",
          "constraint model", "gene-identity null, random split",
          "null, pairs within one missense-Z decile"),
  v = c(vb$mis_z, vb$minus_mis_oe, vb$minus_loeuf, vb$constraint_model_leave_gene_out,
        vb$gene_identity_null_random_split, j$gene_null_within_constraint_deciles),
  col = c(AQUA, AQUA, AQUA, JADE_RAMP[5], NULL_COL, NULL_T)) %>%
  arrange(v) %>%
  mutate(x0 = (0.5 + v) / 2, r = (v - 0.5) / 2, gene = v > 0.8)
cons <- filter(rb, !gene) %>% arrange(desc(v)) %>%
  mutate(ly = c(0.090, 0.068, 0.046, 0.024), lx = 0.684)
genes_r <- filter(rb, gene) %>% arrange(desc(v)) %>% mutate(ly = c(0.262, 0.236))
ticks7 <- tibble(v = seq(0.5, 1, 0.1))
pb <- ggplot() +
  ggforce::geom_arc_bar(data = rb, aes(x0 = x0, y0 = 0, r0 = 0, r = r, start = -pi / 2,
                                       end = pi / 2, fill = I(col)),
                        alpha = 0.07, colour = NA) +
  ggforce::geom_arc(data = rb, aes(x0 = x0, y0 = 0, r = r, start = -pi / 2, end = pi / 2,
                                   colour = I(col), linewidth = I(ifelse(gene, 1.3, 0.9)))) +
  annotate("segment", x = 0.5, xend = 1.0, y = 0, yend = 0, colour = INK2, linewidth = 0.35) +
  geom_segment(data = ticks7, aes(x = v, xend = v, y = 0, yend = -0.008), colour = INK2,
               linewidth = 0.35) +
  geom_text(data = ticks7, aes(v, -0.02, label = sprintf("%.1f", v)), vjust = 1, size = 2.3,
            colour = INK2) +
  annotate("text", x = 0.75, y = -0.055, label = "AUROC", vjust = 1, size = 2.45,
           colour = INK2) +
  geom_point(data = rb, aes(v, 0, fill = I(col)), shape = 21, colour = SURF, size = 2.2,
             stroke = 0.5) +
  geom_segment(data = cons, aes(x = v, y = 0.004, xend = lx - 0.004, yend = ly),
               colour = FAINT, linewidth = 0.25) +
  geom_text(data = cons, aes(lx, ly, label = sprintf("%.3f", v)), hjust = 0, size = 2.3,
            fontface = "bold", colour = INK) +
  geom_text(data = cons, aes(lx + 0.045, ly, label = lab), hjust = 0, size = 1.95,
            colour = INK2) +
  geom_text(data = genes_r, aes(0.712, ly, label = sprintf("%.3f", v)), hjust = 1, size = 2.3,
            fontface = "bold", colour = INK) +
  geom_text(data = genes_r, aes(0.718, ly, label = lab), hjust = 0, size = 2.05,
            colour = INK2) +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  coord_equal(xlim = c(0.485, 1.015), ylim = c(-0.075, 0.28), clip = "off") +
  theme_void(base_family = FAMILY) + theme(plot.margin = margin(18, 8, 6, 8))

fig <- tags(pa + pb + plot_layout(widths = c(1, 0.95)))
save_figure(fig, "Fig7", width = 7.5, height = 3.6)
