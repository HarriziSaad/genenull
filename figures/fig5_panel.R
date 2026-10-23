source("figures/theme_final.R")

t <- read_csv_res("dbnsfp_matched.csv") %>%
  mutate(label = pretty_pred(predictor),
         label = ifelse(label == "MPC", "MPC*", label),
         cls = factor(atlas_class(atlas_type), levels = names(CLASS_COL)),
         col = CLASS_COL[as.character(cls)])
stopifnot(!any(is.na(t$cls)))
j  <- read_json_res("dbnsfp_matched.json")
sw <- read_csv_res("dbnsfp_matched_sweep.csv")
fit <- j$headroom_fit$pooled
signed <- function(x) ifelse(abs(x) < 0.0005, "0.000", minus(sprintf("%+.3f", x)))

GAP <- 1.0
a <- t %>% arrange(cls, desc(change_pooled)) %>%
  mutate(g = as.integer(cls), pos = row_number() + (g - 1) * GAP, yy = max(pos) - pos + 1,
         fade = ifelse(label == "MPC*", 0.45, 1))
heads <- a %>% group_by(cls) %>% summarise(y = max(yy) + 0.75, .groups = "drop")
comet <- a %>% select(label, yy, col, fade, global_matched, within_pooled) %>%
  pivot_longer(c(global_matched, within_pooled), names_to = "end", values_to = "x") %>%
  mutate(w = ifelse(end == "global_matched", 0.15, 2.6),
         al = ifelse(end == "global_matched", 0.05, 1) * fade)
XA <- c(0.745, 0.985)
stopifnot(all(a$global_matched > XA[1], a$within_pooled < XA[2]))
pa <- ggplot(a) +
  ggforce::geom_link2(data = comet, aes(x, yy, group = label, colour = I(col),
                                        linewidth = w, alpha = al), n = 80,
                      lineend = "round") +
  geom_point(aes(within_pooled, yy, colour = I(col), alpha = I(fade)), size = 2.3) +
  geom_point(aes(global_matched, yy), colour = INK2, size = 0.5, alpha = 0.6) +
  geom_text(aes(x = 0.993, y = yy, label = signed(change_pooled)), hjust = 0, size = 2.2,
            colour = INK2) +
  geom_point(data = heads, aes(x = XA[1] + 0.004, y = y,
                               colour = I(CLASS_COL[as.character(cls)])),
             shape = 15, size = 1.9) +
  geom_text(data = heads, aes(x = XA[1] + 0.009, y = y, label = toupper(cls)), hjust = 0,
            size = 1.95, fontface = "bold", colour = INK2) +
  annotate("text", x = 0.993, y = max(a$yy) + 0.75, label = "gain", hjust = 0, size = 2.0,
           colour = MUTED) +
  scale_linewidth_identity() + scale_alpha_identity() +
  scale_y_continuous(breaks = a$yy, labels = a$label, expand = expansion(add = c(0.6, 1.1))) +
  scale_x_continuous(breaks = seq(0.75, 0.95, 0.05)) +
  coord_cartesian(xlim = XA, clip = "off") +
  labs(x = "AUROC: global (tail) to within genes (head)", y = NULL) +
  theme_final() + no_y_grid +
  theme(axis.text.y = element_text(colour = INK, size = 6.9),
        plot.margin = margin(8, 30, 6, 4))

trail <- bind_rows(
  t %>% transmute(predictor, alpha = 0, global = global_matched, change_pooled),
  sw %>% select(predictor, alpha, global, change_pooled)) %>%
  arrange(predictor, alpha)
gx <- seq(min(sw$global), max(t$global_matched), length.out = 250)
curve <- tibble(x = gx, y = fit[1] * gx^2 + fit[2] * gx + fit[3])
above <- bind_rows(curve, tibble(x = rev(range(gx)), y = c(0.2, 0.2)))
pb <- ggplot() +
  geom_polygon(data = above, aes(x, y), fill = WASH) +
  geom_path(data = trail, aes(global, change_pooled, group = predictor), colour = FAINT,
            linewidth = 0.3) +
  geom_point(data = sw, aes(global, change_pooled), colour = FAINT, size = 0.25) +
  geom_path(data = curve, aes(x, y), colour = INK, linewidth = 0.55, linetype = "22") +
  geom_hline(yintercept = 0, colour = INK2, linewidth = 0.3) +
  geom_point(data = t, aes(global_matched, change_pooled, fill = I(col)), shape = 21,
             colour = SURF, size = 2.0, stroke = 0.5) +
  geom_text_repel(data = t, aes(global_matched, change_pooled, label = label), size = 1.95,
                  colour = INK, family = FAMILY, segment.colour = MUTED, segment.size = 0.2,
                  min.segment.length = 0, box.padding = 0.18, point.padding = 0.12,
                  max.overlaps = Inf, seed = 4, force = 6, force_pull = 0.6,
                  max.time = 4, max.iter = 100000, xlim = c(0.725, 0.977),
                  ylim = c(-0.048, 0.15)) +
  annotate("text", x = 0.975, y = 0.147, hjust = 1, vjust = 1, size = 2.15, colour = INK2,
           lineheight = 0.95, label = "more gain than reduced\naccuracy alone produces") +
  annotate("text", x = 0.724, y = fit[1] * 0.724^2 + fit[2] * 0.724 + fit[3] + 0.004,
           hjust = 0, vjust = 0, size = 2.15, colour = INK, label = "accuracy alone") +
  annotate("text", x = 0.727, y = -0.046, hjust = 0, vjust = 0, size = 2.0, colour = MUTED,
           lineheight = 0.95,
           label = "grey trails: each predictor's scores mixed\nwith noise in steps of 5%") +
  scale_x_continuous(breaks = seq(0.75, 0.95, 0.05)) +
  scale_y_continuous(labels = function(v) minus(sprintf("%+.2f", v)),
                     breaks = seq(-0.04, 0.16, 0.04)) +
  coord_cartesian(xlim = c(0.72, 0.979), ylim = c(-0.05, 0.152), expand = FALSE,
                  clip = "on") +
  labs(x = "Global AUROC on the same variants", y = "Within-gene minus global AUROC") +
  theme_final() + theme(plot.margin = margin(8, 8, 6, 6))

S <- j$schemes
SCHEME <- tribble(
  ~scheme,                        ~cls,                      ~ens,
  "submitted",                    "Plan, as submitted",       "not applied",
  "submitted_high_confidence",    "Plan, high confidence",    "not applied",
  "amended",                      "Plan, corrected",          "MPC",
  "amended_ensemble_named",       "Plan, corrected",          "named in the Atlas",
  "amended_ensemble_named_plus",  "Plan, corrected",          "named, plus VARITY, MutScore",
  "amended_ensemble_flagged",     "Plan, corrected",          "every one the Atlas flags",
  "atlas",                        "Atlas classes",            "MPC",
  "atlas_misfit_as_tuned",        "Atlas, MisFit as tuned",   "MPC",
  "atlas_ensemble_named",         "Atlas classes",            "named in the Atlas",
  "atlas_ensemble_named_plus",    "Atlas classes",            "named, plus VARITY, MutScore",
  "atlas_excluding_meta",         "Atlas classes",            "every one the Atlas flags")
METRIC <- tribble(
  ~key,                    ~stat,     ~corr,
  "did_pooled",            "pooled",  "none",
  "did_mean",              "mean",    "none",
  "did_pooled_headroom",   "pooled",  "degradation control")
spec <- tidyr::crossing(SCHEME, METRIC) %>% rowwise() %>%
  mutate(est = S[[scheme]][[key]]$did, lo = S[[scheme]][[key]]$boot_both$ci[1],
         hi = S[[scheme]][[key]]$boot_both$ci[2], p = S[[scheme]][[key]]$boot_both$p) %>%
  ungroup() %>% arrange(est) %>% mutate(xx = row_number(), sig = p < 0.05,
         main = (scheme %in% c("submitted", "atlas")) & key == "did_pooled")
stopifnot(nrow(spec) == 33)
LEV <- list("Classification" = c("Plan, as submitted", "Plan, high confidence",
                                 "Plan, corrected", "Atlas classes", "Atlas, MisFit as tuned"),
            "Ensembles left out" = c("not applied", "MPC", "named in the Atlas",
                                     "named, plus VARITY, MutScore",
                                     "every one the Atlas flags"),
            "Within-gene AUROC" = c("pooled", "mean"),
            "Correction" = c("none", "degradation control"))
grid_rows <- bind_rows(lapply(names(LEV), function(g) tibble(group = g, level = LEV[[g]]))) %>%
  mutate(yy = -(row_number() + match(group, names(LEV)) * 1.05))
marks <- bind_rows(
  spec %>% transmute(xx, sig, group = "Classification", level = cls),
  spec %>% transmute(xx, sig, group = "Ensembles left out", level = ens),
  spec %>% transmute(xx, sig, group = "Within-gene AUROC", level = stat),
  spec %>% transmute(xx, sig, group = "Correction", level = corr)) %>%
  left_join(grid_rows, by = c("group", "level"))
stopifnot(!any(is.na(marks$yy)))
gheads <- grid_rows %>% group_by(group) %>% summarise(y = max(yy) + 0.95, .groups = "drop")
XS <- c(-5.6, nrow(spec) + 0.6)
pc_top <- ggplot(spec) +
  geom_hline(yintercept = 0, colour = INK2, linewidth = 0.3) +
  geom_segment(aes(x = xx, xend = xx, y = lo, yend = hi, colour = I(ifelse(sig, CONTRAST, FAINT))),
               linewidth = 0.55) +
  geom_point(data = filter(spec, main), aes(xx, est), shape = 21, size = 3.6, stroke = 0.5,
             colour = INK, fill = NA) +
  geom_point(aes(xx, est, fill = I(ifelse(sig, CONTRAST, SURF))), shape = 21, size = 1.7,
             stroke = 0.5, colour = CONTRAST) +
  annotate("text", x = XS[1] + 0.2, y = 0.085, hjust = 0, vjust = 1, size = 2.1, colour = INK,
           lineheight = 0.95,
           label = "difference in gain,\nnot clinically trained\nminus clinically trained") +
  annotate("text", x = XS[1] + 0.2, y = 0.027, hjust = 0, vjust = 1, size = 2.0, colour = MUTED,
           lineheight = 0.95, label = "filled: p < 0.05\nringed: in the abstract") +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(labels = function(v) minus(sprintf("%+.2f", v)),
                     breaks = seq(-0.02, 0.08, 0.02)) +
  coord_cartesian(xlim = XS, ylim = c(-0.022, 0.09), clip = "off") +
  labs(x = NULL, y = NULL) +
  theme_final() + no_x_grid + theme(axis.text.x = element_blank(),
                                    plot.margin = margin(4, 8, 0, 6))
pc_grid <- ggplot() +
  geom_segment(data = grid_rows, aes(x = 0.5, xend = nrow(spec) + 0.5, y = yy, yend = yy),
               colour = GRID, linewidth = 0.3) +
  geom_point(data = marks, aes(xx, yy, colour = I(ifelse(sig, CONTRAST, MUTED))), size = 1.05) +
  geom_text(data = grid_rows, aes(XS[1] + 0.2, yy, label = level), hjust = 0, size = 1.95,
            colour = INK2) +
  geom_text(data = gheads, aes(XS[1] + 0.2, y, label = group), hjust = 0, size = 2.05,
            colour = INK, fontface = "bold") +
  scale_x_continuous(expand = c(0, 0)) +
  coord_cartesian(xlim = XS, clip = "off") +
  theme_void(base_family = FAMILY) + theme(plot.margin = margin(0, 8, 4, 6))
pc <- pc_top / pc_grid + plot_layout(heights = c(1, 1.35))

top <- (pa | pb) + plot_layout(widths = c(0.92, 1.08))
fig <- (top / pc + plot_layout(heights = c(1.9, 1.35))) +
  plot_annotation(tag_levels = list(c("a", "b", "c", ""))) &
  theme(plot.tag = element_text(face = "bold", size = 11, colour = INK, family = FAMILY))
save_figure(fig, "Fig5", width = 7.5, height = 8.6)
