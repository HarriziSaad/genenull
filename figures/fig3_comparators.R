source("figures/theme_final.R")

hh <- read_csv_res("head_to_head.csv")
KEYS <- c("varity_r_full", "varity", "gmvp", "alphamissense")
nul <- filter(hh, column == "gene_prior")
d <- hh %>% filter(column %in% KEYS) %>%
  mutate(name = c("VARITY, standard score*", "VARITY, leave-one-out", "gMVP",
                  "AlphaMissense")[match(column, KEYS)],
         cap = PRED_COL[c("VARITY", "VARITY", "gMVP", "AlphaMissense")[match(column, KEYS)]],
         hollow = column == "varity_r_full") %>%
  arrange(auroc) %>% mutate(ring = row_number() + 1.15)
TURN <- 0.75
XMAX <- 0.5 + 0.5 / TURN
H <- 0.34
ticks <- tibble(v = seq(0.6, 1.0, 0.1))
R_OUT <- max(d$ring) + 0.55

p <- ggplot() +
  geom_segment(data = ticks, aes(x = v, xend = v, y = 0.62, yend = R_OUT), colour = GRID,
               linewidth = 0.35) +
  geom_text(data = ticks, aes(v, R_OUT + 0.42, label = sprintf("%.1f", v)), size = 2.3,
            colour = MUTED) +
  annotate("text", x = 0.5, y = R_OUT + 0.42, label = "0.5", size = 2.3, colour = MUTED) +
  annotate("rect", xmin = 0.5, xmax = nul$auroc, ymin = 1 - H, ymax = 1 + H,
           fill = NULL_COL) +
  annotate("segment", x = nul$ci_lo, xend = nul$ci_hi, y = 1 + H + 0.12, yend = 1 + H + 0.12,
           colour = INK, linewidth = 0.45) +
  geom_rect(data = d, aes(xmin = 0.5, xmax = nul$auroc, ymin = ring - H, ymax = ring + H),
            fill = NULL_T) +
  geom_rect(data = filter(d, !hollow), aes(xmin = nul$auroc, xmax = auroc, ymin = ring - H,
                                          ymax = ring + H, fill = I(cap))) +
  geom_rect(data = filter(d, hollow), aes(xmin = nul$auroc + 0.0015, xmax = auroc,
                                         ymin = ring - H + 0.05, ymax = ring + H - 0.05),
            fill = SURF, colour = ORANGE, linewidth = 0.55) +
  geom_segment(data = d, aes(x = ci_lo, xend = ci_hi, y = ring + H + 0.12,
                             yend = ring + H + 0.12), colour = INK, linewidth = 0.45) +
  annotate("segment", x = nul$auroc, xend = nul$auroc, y = 1 - H - 0.1, yend = R_OUT - 0.2,
           colour = NULL_DARK, linewidth = 0.35, linetype = "22") +
  geom_richtext(data = d, aes(0.5, ring, label = sprintf(
    "%s <span style='color:%s'>%.3f</span> <b>+%.3f</b>", name, MUTED, auroc,
    margin_over_null)), hjust = 1.04, size = 2.35, colour = INK, fill = NA,
    label.colour = NA, label.padding = unit(0, "pt"), family = FAMILY) +
  annotate("richtext", x = 0.5, y = 1, hjust = 1.04, size = 2.35, fill = NA,
           label.colour = NA, label.padding = unit(0, "pt"), family = FAMILY,
           label = sprintf("<b style='color:%s'>gene-identity null</b> <b>%.3f</b>",
                           NULL_COL, nul$auroc)) +
  annotate("text", x = 0.5, y = -2.4, size = 2.15, colour = INK2, lineheight = 0.95,
           label = sprintf("AUROC on\n%s variants", comma(nul$n))) +
  scale_fill_identity() +
  scale_x_continuous(limits = c(0.5, XMAX), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-2.4, R_OUT + 0.6), expand = c(0, 0)) +
  coord_polar(theta = "x", start = 0, direction = 1, clip = "off") +
  theme_void(base_family = FAMILY) + theme(plot.margin = margin(4, 4, 4, 4))
save_figure(p, "Fig3", width = 5.4, height = 5.0)
