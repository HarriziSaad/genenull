source("figures/theme_final.R")
suppressPackageStartupMessages(library(geomtextpath))

sp  <- read_json_res("split_scheme_report.json")
gpr <- read_csv_res("gene_prior_robustness.csv")
cmp <- read_csv_res("compartment_generalization.csv")
roc <- read_fdata("roc_curves.csv")
chn <- read_fdata("null_chain.csv")
pur <- read_fdata("purity.csv")
cj  <- fromJSON(file.path(FDATA, "null_chain.json"))
psq <- fromJSON(file.path(FDATA, "pair_square.json"))

path_pts <- bind_rows(tibble(x = 0, y = 0), transmute(chn, x = x1, y = y1))
under <- bind_rows(path_pts, tibble(x = c(1, 0), y = c(0, 0)))
am <- filter(roc, curve == "alphamissense")
am_area <- bind_rows(transmute(am, x = fpr, y = tpr), tibble(x = c(1, 0), y = c(0, 0)))
rnd <- filter(roc, curve == "null_random") %>% transmute(x = fpr, y = tpr)
mark <- lapply(c(0.9, 0.5, 0.1), function(f) {
  r <- chn[which(chn$frac < f)[1], ]
  tibble(f = f, x = r$x0, y = r$y0)
}) %>% bind_rows()
win <- c(x0 = 0.148, x1 = 0.168, y0 = 0.879, y1 = 0.899)
pure_b_x <- 1 - cj$benign_in_pure_benign_genes

am_at <- function(xv) approx(am$fpr, am$tpr, xout = xv, ties = max)$y
null_at <- function(xv) approx(path_pts$x, path_pts$y, xout = xv, ties = max)$y
am_lab_y <- (am_at(0.035) + null_at(0.035)) / 2
pa <- ggplot() +
  geom_polygon(data = am_area, aes(x, y), fill = AQUA) +
  geom_polygon(data = under, aes(x, y), fill = null_gradient()) +
  geom_path(data = rnd, aes(x, y), colour = SURF, linetype = "22", linewidth = 0.4) +
  annotate("text", x = 0.76, y = 0.935, size = 2.2, colour = SURF,
           label = sprintf("random split %.3f", sp$gene_prior_randomsplit$auroc)) +
  annotate("segment", x = 0, y = 0, xend = 1, yend = 1, colour = SURF, linetype = "13",
           linewidth = 0.45) +
  annotate("text", x = 0.47, y = 0.49, angle = 45, hjust = 0.5, vjust = 0, size = 2.2,
           colour = SURF, label = "leave-gene-out 0.500") +
  geom_path(data = path_pts, aes(x, y), colour = NULL_DARK, linewidth = 0.35) +
  geom_point(data = mark, aes(x, y), shape = 21, size = 1.5, stroke = 0.6,
             fill = SURF, colour = INK) +
  annotate("text", x = mark$x[1] + 0.024, y = mark$y[1] - 0.004, hjust = 0, vjust = 1,
           size = 2.1, colour = SURF, label = "null score 0.9") +
  annotate("text", x = mark$x[2] + 0.024, y = mark$y[2] - 0.004, hjust = 0, vjust = 1,
           size = 2.1, colour = SURF, label = "0.5") +
  annotate("text", x = mark$x[3], y = 1.022, hjust = 0.5, vjust = 0,
           size = 2.1, colour = INK2, label = "0.1") +
  annotate("text", x = 0.70, y = 0.27, hjust = 0.5, size = 3.0, colour = SURF,
           fontface = "bold", label = sprintf("%.3f", sp$gene_prior_insample$auroc)) +
  annotate("text", x = 0.70, y = 0.225, hjust = 0.5, size = 2.15, colour = SURF,
           label = "gene-identity null, in-sample") +
  annotate("segment", x = pure_b_x, xend = 1, y = 1.035, yend = 1.035,
           colour = BENIGN, linewidth = 2.2) +
  annotate("segment", x = pure_b_x, xend = pure_b_x, y = 1, yend = 1.05,
           colour = INK2, linewidth = 0.25) +
  annotate("text", x = 1, y = 1.075, hjust = 1, vjust = 0, size = 2.15, colour = INK,
           lineheight = 0.95,
           label = sprintf("%s of benign variants are in genes\nwith no pathogenic variant",
                           percent(cj$benign_in_pure_benign_genes, accuracy = 0.1))) +
  annotate("segment", x = 0.035, xend = 0.035, y = am_lab_y, yend = 1.075,
           colour = INK2, linewidth = 0.25) +
  annotate("point", x = 0.035, y = am_lab_y, size = 0.7, colour = INK) +
  annotate("text", x = 0.045, y = 1.075, hjust = 0, vjust = 0, size = 2.15, colour = INK,
           label = sprintf("AlphaMissense %.3f", sp$alphamissense$auroc)) +
  annotate("rect", xmin = win[["x0"]], xmax = win[["x1"]], ymin = win[["y0"]],
           ymax = win[["y1"]], fill = NA, colour = INK, linewidth = 0.5) +
  annotate("text", x = win[["x1"]] + 0.012, y = win[["y1"]] + 0.012, hjust = 0, vjust = 0,
           size = 2.4, colour = INK, fontface = "bold", label = "b") +
  scale_x_continuous(breaks = seq(0, 1, 0.25), labels = c("0", "0.25", "0.5", "0.75", "1"),
                     expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(0, 1, 0.25), labels = c("0", "0.25", "0.5", "0.75", "1"),
                     expand = c(0, 0)) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1), clip = "off") +
  labs(x = "False positive rate", y = "True positive rate") +
  theme_final() + no_grid + theme(plot.margin = margin(30, 10, 6, 14))

wg <- chn %>% filter(x1 > win[["x0"]], x0 < win[["x1"]], y1 > win[["y0"]], y0 < win[["y1"]])
wpath <- bind_rows(tibble(x = wg$x0[1], y = wg$y0[1]), transmute(wg, x = x1, y = y1))
wunder <- bind_rows(wpath, tibble(x = c(max(wpath$x), min(wpath$x)),
                                  y = c(win[["y0"]] - 1, win[["y0"]] - 1)))
big <- wg %>% filter(p * b >= 2000, x1 < win[["x1"]] - 0.0035)
mag <- round(1 / (win[["x1"]] - win[["x0"]]))
pb <- ggplot() +
  annotate("rect", xmin = win[["x0"]], xmax = win[["x1"]], ymin = win[["y0"]],
           ymax = win[["y1"]], fill = SURF) +
  geom_polygon(data = wunder, aes(x, y), fill = NULL_COL) +
  geom_rect(data = wg, aes(xmin = x0, xmax = x1, ymin = y0, ymax = y1),
            fill = NULL_T, colour = SURF, linewidth = 0.25, alpha = 0.9) +
  geom_path(data = wpath, aes(x, y), colour = NULL_DARK, linewidth = 0.45) +
  geom_text(data = big, aes(x1 + 0.0004, (y0 + y1) / 2, label = gene_symbol), hjust = 0,
            size = 2.2, colour = INK, fontface = "bold") +
  annotate("text", x = win[["x0"]] + 0.0007, y = win[["y1"]] - 0.0007, hjust = 0, vjust = 1,
           size = 2.2, colour = INK2, label = sprintf("%d\u00d7", mag)) +
  annotate("text", x = win[["x0"]] + 0.0007, y = win[["y1"]] - 0.0032, hjust = 0, vjust = 1,
           size = 2.2, colour = INK, lineheight = 0.95,
           label = sprintf("each rectangle: the pairs within one gene\nall genes together: %s of pairs",
                           percent(psq$same_gene_share, accuracy = 0.01))) +
  coord_equal(xlim = win[c("x0", "x1")], ylim = win[c("y0", "y1")], expand = FALSE) +
  scale_x_continuous(breaks = pretty(win[c("x0", "x1")], 3)) +
  scale_y_continuous(breaks = pretty(win[c("y0", "y1")], 3)) +
  labs(x = "False positive rate", y = "True positive rate") +
  theme_final() + no_grid +
  theme(panel.border = element_rect(fill = NA, colour = INK, linewidth = 0.6),
        plot.margin = margin(30, 10, 6, 10))

cc <- gpr %>% mutate(xx = row_number()) %>% left_join(select(pur, -n_genes), by = "min_variants")
top <- ggplot(cc) +
  geom_ribbon(aes(xx, ymin = auroc, ymax = alphamissense_auroc), fill = AQUA, alpha = 0.35) +
  geom_line(aes(xx, alphamissense_auroc), colour = AQUA, linewidth = 0.7) +
  geom_line(aes(xx, auroc), colour = NULL_COL, linewidth = 0.9) +
  geom_point(aes(xx, auroc), colour = NULL_COL, size = 1.4) +
  geom_point(aes(xx, alphamissense_auroc), colour = AQUA, size = 1.4) +
  geom_text(aes(xx, alphamissense_auroc + 0.006, label = minus(sprintf("+%.3f", gap))),
            size = 2.15, colour = INK, fontface = "bold", vjust = 0) +
  geom_text(aes(xx, auroc - 0.006, label = sprintf("%.3f", auroc)), size = 2.1,
            colour = NULL_COL, vjust = 1) +
  annotate("text", x = 6.45, y = cc$alphamissense_auroc[6], label = "AlphaMissense",
           hjust = 0, size = 2.2, colour = AQUA) +
  annotate("text", x = 6.45, y = cc$auroc[6], label = "null,\nrandom split", hjust = 0,
           size = 2.2, colour = NULL_COL, lineheight = 0.9) +
  scale_x_continuous(limits = c(0.55, 8.15), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.905, 0.975), breaks = c(0.92, 0.94, 0.96),
                     expand = c(0, 0)) +
  labs(x = NULL, y = "AUROC") +
  theme_final() + no_x_grid + theme(axis.text.x = element_blank(),
                                    plot.margin = margin(28, 6, 0, 6))
waf <- cc %>% rowwise() %>% do({
  r <- .
  npb <- round(100 * r$pure_benign); npp <- round(100 * r$pure_pathogenic)
  kind <- c(rep("pb", npb), rep("pp", npp), rep("mix", 100 - npb - npp))
  i <- 0:99
  tibble(xx = r$xx, col = i %% 10, row = i %/% 10, kind = kind)
}) %>% ungroup()
CELL <- 0.074
waf <- waf %>% mutate(cx = xx + (col - 4.5) * CELL, cy = (row - 4.5) * CELL)
bottom <- ggplot(waf) +
  geom_tile(aes(cx, cy, fill = kind), width = CELL * 0.82, height = CELL * 0.82) +
  scale_fill_manual(values = c(pb = BENIGN, pp = PATH, mix = WASH2)) +
  geom_text(data = cc, aes(xx, -0.45, label = percent(pure_benign + pure_pathogenic,
                                                          accuracy = 1)),
            size = 2.2, colour = INK, fontface = "bold", vjust = 1) +
  geom_text(data = cc, aes(xx, -0.66, label = sprintf("\u2265%d", min_variants)),
            size = 2.3, colour = INK2, vjust = 1) +
  geom_text(data = cc, aes(xx, -0.86, label = comma(n_genes)), size = 2.0,
            colour = MUTED, vjust = 1) +
  annotate("text", x = 6.45, y = 0.2, hjust = 0, size = 2.15, colour = INK, lineheight = 0.95,
           label = "label-pure genes\n(each square 1%)") +
  annotate("text", x = 6.45, y = -0.66, hjust = 0, vjust = 1, size = 2.15, colour = INK2,
           label = "variants per gene") +
  annotate("text", x = 6.45, y = -0.86, hjust = 0, vjust = 1, size = 2.0, colour = MUTED,
           label = "genes") +
  scale_x_continuous(limits = c(0.55, 8.15), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-1.02, 0.40), expand = c(0, 0)) +
  coord_equal(clip = "off") +
  theme_void(base_family = FAMILY) + theme(legend.position = "none",
                                          plot.margin = margin(2, 6, 4, 6))
pc <- top / bottom + plot_layout(heights = c(1, 0.40))

dd <- cmp %>% arrange(desc(gene_prior)) %>% mutate(yy = row_number(),
  lab = sprintf("%s <span style='color:%s'>%s</span>", compartment, MUTED, comma(n)))
flow <- dd %>% select(yy, gene_prior, alphamissense) %>%
  pivot_longer(c(gene_prior, alphamissense), names_to = "end", values_to = "x") %>%
  mutate(col = ifelse(end == "gene_prior", NULL_COL, AQUA))
pd <- ggplot(dd) +
  geom_segment(aes(x = 0.80, xend = 1.0, y = yy, yend = yy), colour = GRID, linewidth = 0.3) +
  ggforce::geom_link2(data = flow, aes(x, yy, colour = col, group = yy), linewidth = 2.6,
                      lineend = "round", n = 60) +
  geom_point(aes(gene_prior, yy), colour = NULL_COL, size = 2.2) +
  geom_point(aes(alphamissense, yy), colour = AQUA, size = 2.2) +
  geom_text(aes(alphamissense + 0.007, yy, label = sprintf("+%.3f", gap)), hjust = 0,
            size = 2.1, colour = INK2) +
  geom_richtext(aes(0.795, yy, label = lab), hjust = 1, size = 2.2, fill = NA,
                label.colour = NA, label.padding = unit(0, "pt"), family = FAMILY) +
  annotate("text", x = min(dd$gene_prior), y = max(dd$yy) + 0.9, hjust = 0.5, size = 2.15,
           colour = NULL_COL, label = "null") +
  annotate("text", x = dd$alphamissense[which.min(dd$gene_prior)], y = max(dd$yy) + 0.9,
           hjust = 0.5, size = 2.15, colour = AQUA, label = "AlphaMissense") +
  scale_colour_identity() +
  scale_x_continuous(limits = c(0.66, 1.04), breaks = seq(0.80, 1.0, 0.05),
                     labels = c("0.80", "0.85", "0.90", "0.95", "1.00"), expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.4, max(dd$yy) + 1.3), expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  labs(x = "AUROC (in-sample null and AlphaMissense)", y = NULL) +
  theme_final() + no_grid +
  theme(axis.text.y = element_blank(), plot.margin = margin(28, 10, 6, 6),
        panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3))

fig <- (pa | pb) / (pc | pd) + plot_layout(heights = c(1, 1.05)) +
  plot_annotation(tag_levels = list(c("a", "b", "c", "", "d"))) &
  theme(plot.tag = element_text(face = "bold", size = 11, colour = INK, family = FAMILY))
save_figure(fig, "Fig2", width = 7.5, height = 7.6)
