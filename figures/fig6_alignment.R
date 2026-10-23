source("figures/theme_final.R")

o  <- read_json_res("offset_robustness.json")
ts <- read_json_res("temporal_split.json")

pp <- as_tibble(o$per_predictor) %>%
  mutate(label = pretty_pred(predictor),
         label = ifelse(label == "MPC", "MPC*", label),
         cls = factor(atlas_class(atlas_type), levels = names(CLASS_COL)),
         col = CLASS_COL[as.character(cls)],
         T1 = unlist(ts$T1_new_labels$per_predictor)[predictor],
         T2 = unlist(ts$T2_historical_composition$per_predictor)[predictor])
stopifnot(!any(is.na(pp$cls)), !any(is.na(pp$T1)), !any(is.na(pp$T2)))

CHECKS <- tribble(
  ~key,                                ~lab,                       ~grp,
  "align_adj",                         "Primary (\u22653 each)",      "",
  "align_adj_ge5",                     "\u22655 each",              "THRESHOLDS",
  "align_adj_ge10",                    "\u226510 each",             "THRESHOLDS",
  "align_benign",                      "Benign only",               "LABELS",
  "align_adj_2star",                   "\u22652 stars",              "LABELS",
  "align_adj_partial_constraint",      "Constraint",                "ADJUSTED FOR",
  "align_adj_partial_constraint_cds",  "+ CDS length",              "ADJUSTED FOR",
  "align_adj_partial_density",         "Variant density",           "ADJUSTED FOR",
  "align_adj_partial",                 "Five covariates",           "ADJUSTED FOR",
  "T1",                                "New labels",                "TIME SPLIT",
  "T2",                                "2023 gene mix",             "TIME SPLIT") %>%
  mutate(gi = match(grp, unique(grp)), x = row_number() + (gi - 1) * 0.45)

rows <- pp %>% arrange(cls, desc(align_adj)) %>%
  mutate(g = as.integer(cls), pos = row_number() + (g - 1) * 0.6, yy = max(pos) - pos + 1)
long <- rows %>% select(label, cls, col, yy, all_of(CHECKS$key)) %>%
  pivot_longer(all_of(CHECKS$key), names_to = "key", values_to = "v") %>%
  left_join(CHECKS, by = "key")

LIM <- c(-0.2, 0.45)
stopifnot(all(long$v > LIM[1], long$v < LIM[2]))
NEG <- NEG_GREY
SMAX <- 5.6
size_of <- function(v) SMAX * sqrt(pmin(abs(v), LIM[2]) / LIM[2])
long <- long %>% mutate(sz = size_of(v), fill = ifelse(v >= 0, NULL_COL, NEG))

top <- max(rows$yy)
X0 <- min(CHECKS$x) - 0.5; X1 <- max(CHECKS$x) + 0.5
XL <- c(-4.9, X1 + 0.1)
grp_lab <- CHECKS %>% filter(grp != "") %>% group_by(grp) %>%
  summarise(x0 = min(x) - 0.42, x1 = max(x) + 0.42, .groups = "drop")

pa <- ggplot() +
  annotate("rect", xmin = CHECKS$x[1] - 0.47, xmax = CHECKS$x[1] + 0.47, ymin = 0.45,
           ymax = top + 0.55, fill = WASH2) +
  geom_segment(data = CHECKS, aes(x = x, xend = x, y = 0.5, yend = top + 0.5),
               colour = GRID, linewidth = 0.25) +
  geom_segment(data = rows, aes(x = X0 + 0.1, xend = X1 - 0.1, y = yy, yend = yy),
               colour = GRID, linewidth = 0.25) +
  geom_text(data = CHECKS, aes(x - 0.15, top + 0.75, label = lab), angle = 45, hjust = 0,
            vjust = 0, size = 2.0) +
  geom_segment(data = grp_lab, aes(x = x0, xend = x1, y = top + 3.85, yend = top + 3.85),
               colour = FAINT, linewidth = 0.4) +
  geom_text(data = grp_lab, aes((x0 + x1) / 2, top + 4.0, label = grp), vjust = 0, size = 1.85,
            colour = MUTED, fontface = "bold") +
  geom_point(data = long, aes(x, yy, size = I(sz), colour = I(fill)), shape = 16) +
  geom_point(data = rows, aes(x = X0 - 0.3, y = yy, colour = I(col)), shape = 15, size = 1.7) +
  geom_text(data = rows, aes(x = X0 - 0.62, y = yy, label = ifelse(abs(align_adj) < 0.005, "0.00",
                                                minus(sprintf("%.2f", align_adj)))),
            hjust = 1, size = 2.1, colour = INK2) +
  geom_text(data = rows, aes(x = X0 - 1.75, y = yy, label = label,
                             colour = I(ifelse(label == "MPC*", MUTED, INK))),
            hjust = 1, size = 2.35) +
  scale_x_continuous(limits = XL, expand = c(0, 0)) +
  scale_y_continuous(limits = c(0.4, top + 4.7), expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  theme_void(base_family = FAMILY) + theme(plot.margin = margin(4, 18, 0, 6))

am_val <- rows$align_adj[rows$label == "AlphaMissense"]
cal <- tibble(v = c(o$calibration[["uniform noise"]], am_val,
                    o$calibration[["AlphaMissense + 0.1 x gene pathogenic fraction"]],
                    o$calibration[["AlphaMissense + 0.3 x gene pathogenic fraction"]]),
              lab = c("uniform noise", "AlphaMissense",
                      "AlphaMissense\n+ 0.1 \u00d7 gene fraction",
                      "AlphaMissense\n+ 0.3 \u00d7 gene fraction")) %>%
  mutate(y = 0.62 - (row_number() - 1) * 0.135)
cls_key <- tibble(cls = names(CLASS_COL), col = unname(CLASS_COL),
                  y = 1.12 - 0.07 * (seq_along(CLASS_COL) - 1))
keys <- ggplot() +
  annotate("text", x = 0, y = 1.2, hjust = 0, size = 2.4, fontface = "bold",
           label = "Atlas class") +
  geom_point(data = cls_key, aes(x = 0.05, y = y, colour = I(col)), shape = 15, size = 2.2) +
  geom_text(data = cls_key, aes(x = 0.14, y = y, label = cls), hjust = 0, size = 2.35) +
  annotate("text", x = 0, y = 0.74, hjust = 0, size = 2.4, fontface = "bold",
           label = "Alignment, as bubble area") +
  geom_point(data = cal, aes(x = 0.12, y = y, size = I(size_of(v))), colour = NULL_COL,
             shape = 16) +
  geom_text(data = cal, aes(x = 0.30, y = y, label = minus(sprintf("%+.2f", v))), hjust = 0,
            size = 2.15, colour = INK, fontface = "bold") +
  geom_text(data = cal, aes(x = 0.53, y = y, label = lab), hjust = 0, size = 2.1,
            lineheight = 0.9, colour = INK2) +
  annotate("point", x = 0.12, y = 0.62 - 4 * 0.135, size = size_of(0.1), colour = NEG,
           shape = 16) +
  annotate("text", x = 0.30, y = 0.62 - 4 * 0.135, hjust = 0, size = 2.1, colour = INK2,
           label = "grey: below zero") +
  scale_x_continuous(limits = c(0, 1.3), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.02, 1.26), expand = c(0, 0)) +
  coord_cartesian(clip = "off") +
  theme_void(base_family = FAMILY) + theme(plot.margin = margin(6, 6, 6, 14))

pt <- o$primary_test
rg <- o$robustness_group_tests
grp_val <- function(key) {
  switch(key,
         align_adj = pt$align_adj, align_adj_ge5 = pt$align_adj_ge5,
         align_adj_ge10 = pt$align_adj_ge10, align_benign = pt$align_benign,
         T1 = ts$T1_new_labels, T2 = ts$T2_historical_composition, rg[[key]])
}
diffs <- CHECKS %>% rowwise() %>%
  mutate(d = grp_val(key)$clinical_minus_rest, p = grp_val(key)$p_exact) %>% ungroup() %>%
  mutate(plab = ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
stopifnot(!any(is.na(diffs$d)))
pb <- ggplot(diffs) +
  geom_segment(aes(x = x, xend = x, y = 0, yend = d), colour = CONTRAST, linewidth = 1.1) +
  geom_point(aes(x, d), colour = CONTRAST, size = 2.0) +
  geom_text(aes(x, d + 0.02, label = plab), vjust = 0, size = 1.95, colour = INK2) +
  annotate("text", x = XL[1] + 0.1, y = 0.2, hjust = 0, vjust = 1, size = 2.3, lineheight = 1.0,
           label = "Clinically trained\nminus the rest") +
  annotate("text", x = XL[1] + 0.1, y = 0.08, hjust = 0, vjust = 1, size = 2.0, colour = MUTED,
           lineheight = 1.0, label = "numbers: exact\npermutation p") +
  geom_hline(yintercept = 0, colour = INK, linewidth = 0.4) +
  scale_x_continuous(limits = XL, expand = c(0, 0)) +
  scale_y_continuous(breaks = c(0, 0.1, 0.2), limits = c(0, 0.29), expand = c(0, 0),
                     position = "right") +
  labs(x = NULL, y = NULL) +
  theme_final() + no_x_grid + theme(axis.text.x = element_blank(),
                                    plot.margin = margin(0, 6, 6, 6))

sc <- rows %>% transmute(label, cls, col, full = align_adj, new = T1)
rho_c <- cor(sc$full, sc$new, method = "spearman")
lab_c <- tribble(
  ~label,          ~dx,     ~dy,     ~hj,
  "MutFormer",      0.008,  -0.022,   1,
  "MVP",            0,      -0.022,   0.5,
  "VARITY",         0,      -0.021,   0.5,
  "AlphaMissense", -0.038,   0.03,    0.5,
  "popEVE",         0,      -0.022,   0.5) %>% inner_join(sc, by = "label")
stopifnot(nrow(lab_c) == 5)
XC <- c(-0.07, 0.31); YC <- c(-0.11, 0.25)
stopifnot(all(sc$full > XC[1], sc$full < XC[2], sc$new > YC[1], sc$new < YC[2]))
pc <- ggplot(sc, aes(full, new)) +
  annotate("polygon", x = c(XC[1], YC[2], XC[2], XC[2], XC[1]),
           y = c(XC[1], YC[2], YC[2], YC[1], YC[1]), fill = WASH) +
  geom_hline(yintercept = 0, colour = GRID, linewidth = 0.5) +
  geom_vline(xintercept = 0, colour = GRID, linewidth = 0.5) +
  geom_abline(slope = 1, intercept = 0, colour = INK2, linetype = "42", linewidth = 0.45) +
  geom_point(aes(fill = I(col), alpha = I(ifelse(label == "MPC*", 0.5, 1))), shape = 21,
             colour = SURF, stroke = 0.7, size = 2.8) +
  geom_text(data = lab_c, aes(full + dx, new + dy, label = label, hjust = hj), size = 2.1) +
  annotate("text", x = XC[1] + 0.005, y = YC[2] - 0.005, hjust = 0, vjust = 1, size = 2.3,
           label = minus(sprintf("Spearman \u03c1 = %+.2f", rho_c))) +
  annotate("text", x = XC[2] - 0.005, y = YC[1] + 0.006, hjust = 1, vjust = 0, size = 2.0,
           colour = MUTED, label = "smaller on new variants") +
  scale_x_continuous(breaks = seq(0, 0.3, 0.1), labels = function(x) minus(sprintf("%+.1f", x)),
                     expand = c(0, 0)) +
  scale_y_continuous(breaks = seq(-0.1, 0.2, 0.1), labels = function(x) minus(sprintf("%+.1f", x)),
                     expand = c(0, 0)) +
  coord_equal(xlim = XC, ylim = YC, clip = "off") +
  labs(x = "Alignment, full table", y = "Alignment, variants\nclassified after 2023") +
  theme_final()

fig <- ((pa / pb + plot_layout(heights = c(1, 0.2))) |
          (keys / pc + plot_layout(heights = c(1.95, 1)))) +
  plot_layout(widths = c(2.05, 1)) +
  plot_annotation(tag_levels = list(c("a", "b", "", "c"))) &
  theme(plot.tag = element_text(face = "bold", size = 11, colour = INK, family = FAMILY))
save_figure(fig, "Fig6", width = 7.5, height = 8.0)
