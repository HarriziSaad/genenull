source("figures/theme_final.R")

mc  <- read_json_res("mito_case_study.json")
abl <- read_csv_res("ablation.csv")
pg  <- read_csv_res("proteingym_results.csv")

nice <- c(alphamissense = "AlphaMissense", varity = "VARITY", gmvp = "gMVP",
          esm2_score = "ESM-2 zero-shot", gene_prior_insample = "Null, in-sample",
          gene_prior_randomsplit = "Null, random split")
LIGHT <- "#AEB2BC"

fa <- bind_rows(lapply(c("all", "2star"), function(tg) {
  v <- mc$mito_minus_control[[tg]]
  bind_rows(lapply(names(v), function(k) tibble(
    key = k, set = tg, d = v[[k]]$difference, lo = v[[k]]$ci[1], hi = v[[k]]$ci[2],
    p = v[[k]]$p)))
})) %>% filter(key %in% names(nice))
ord <- names(nice)[names(nice) %in% fa$key]
fa <- fa %>% mutate(row = match(key, rev(ord)),
                    yy = row + ifelse(set == "all", 0.18, -0.18),
                    fill = ifelse(set == "all", LIGHT, INK),
                    plab = ifelse(p < 0.05, sprintf("p = %.3f", p), ""))
XA <- c(-0.085, 0.06)
stopifnot(all(fa$lo > XA[1], fa$hi < XA[2]))
KEY_A <- c("All variants" = LIGHT, "\u22652 review stars" = INK)
pa <- ggplot(fa) +
  geom_rect(aes(xmin = 0, xmax = d, ymin = yy - 0.15, ymax = yy + 0.15, fill = fill)) +
  geom_segment(aes(x = lo, xend = hi, y = yy, yend = yy), colour = INK, linewidth = 0.35) +
  geom_vline(xintercept = 0, colour = INK, linewidth = 0.4) +
  geom_text(aes(x = XA[2], y = yy, label = plab), hjust = 1, size = 2.2, colour = INK2) +
  scale_fill_identity(guide = guide_legend(nrow = 1), breaks = unname(KEY_A),
                      labels = names(KEY_A)) +
  scale_x_continuous(labels = function(x) minus(sprintf("%+.2f", x)), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq_along(ord), labels = unname(nice[rev(ord)]),
                     expand = expansion(add = 0.5)) +
  coord_cartesian(xlim = XA, clip = "off") +
  labs(x = "Mitochondrial minus matched control (AUROC)", y = NULL) +
  theme_final() + no_y_grid + top_left_legend +
  theme(axis.text.y = element_text(colour = INK, size = 7.2), legend.title = element_blank())

full <- abl$auroc[abl$ablation == "Full interpretable set"]
alone <- abl %>% filter(grepl(" alone$", ablation)) %>%
  transmute(group = sub(" alone$", "", ablation), auroc, kind = "alone")
without <- abl %>% filter(grepl("^minus ", ablation)) %>%
  transmute(group = sub("^minus ", "", ablation), auroc, kind = "without")
lv <- alone$group[order(alone$auroc)]
ab <- bind_rows(alone, without) %>%
  mutate(row = match(group, lv), yy = row + ifelse(kind == "alone", 0.18, -0.18),
         fill = ifelse(kind == "alone", INK2, "#D3D6DE"))
stopifnot(!anyNA(ab$row))
KEY_B <- c("Group alone" = INK2, "All features but the group" = "#D3D6DE")
pb <- ggplot(ab) +
  geom_rect(aes(xmin = 0.5, xmax = auroc, ymin = yy - 0.15, ymax = yy + 0.15, fill = fill)) +
  geom_vline(xintercept = 0.5, colour = INK, linewidth = 0.4) +
  geom_vline(xintercept = full, colour = INK, linewidth = 0.4, linetype = "42") +
  annotate("text", x = full + 0.004, y = length(lv) + 0.62, hjust = 0, size = 2.2,
           colour = INK2, label = sprintf("all 28 features %.3f", full)) +
  scale_fill_identity(guide = guide_legend(nrow = 1), breaks = unname(KEY_B),
                      labels = names(KEY_B)) +
  scale_x_continuous(breaks = seq(0.4, 0.8, 0.1), expand = c(0, 0)) +
  scale_y_continuous(breaks = seq_along(lv), labels = lv, expand = expansion(add = c(0.5, 0.8))) +
  coord_cartesian(xlim = c(0.4, 0.86), clip = "off") +
  labs(x = "AUROC, \u22652-star mitochondrial set", y = NULL) +
  theme_final() + no_y_grid + top_left_legend +
  theme(axis.text.y = element_text(colour = INK, size = 7.2), legend.title = element_blank())

meth <- c("AlphaMissense", "VARITY", "gMVP", "ESM-2 zero-shot")
pc_d <- bind_rows(lapply(meth, function(m) tibble(
  gene = pg$gene, n_sub = pg$n_substitutions, method = m,
  rho = pg[[paste0(m, "_spearman")]], frac = pg[[paste0(m, "_n")]] / pg$n_substitutions))) %>%
  mutate(shown = !is.na(rho) & frac >= 0.5,
         lab = ifelse(shown, sprintf("%.2f", rho), sprintf("%.0f%% scored", 100 * frac)),
         gene_lab = sprintf("%s\n%s substitutions", gene, comma(n_sub)),
         method = factor(method, levels = rev(meth)))
pc <- ggplot(pc_d, aes(gene_lab, method)) +
  geom_tile(aes(fill = ifelse(shown, rho, NA_real_)), colour = SURF, linewidth = 1.6) +
  geom_text(aes(label = lab, colour = I(ifelse(shown & rho > 0.6, SURF, INK)),
                fontface = ifelse(shown, "bold", "plain")), size = 2.6) +
  scale_fill_gradientn(colours = c("#E9EBF1", "#B9BECB", "#7E8597", "#4B5060"),
                       limits = c(0.45, 0.66), na.value = WASH, guide = "none") +
  labs(x = NULL, y = NULL) +
  theme_final() + no_grid +
  theme(axis.text.x = element_text(colour = INK, size = 7.4, lineheight = 1.05),
        axis.text.y = element_text(colour = INK, size = 7.4))

fig <- tags((pa | pb) / pc + plot_layout(heights = c(1, 0.6)))
save_figure(fig, "S1_Fig", width = 7.5, height = 5.0)
