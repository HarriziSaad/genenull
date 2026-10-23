source("figures/theme_final.R")

cal <- read_csv_res("calibration.csv")
cs  <- read_json_res("calibration_summary.json")
pu  <- read_csv_res("protein_uncertainty.csv")

KEYS <- c("varity", "gmvp", "alphamissense", "esm2_score")
NAMES <- c(varity = "VARITY", gmvp = "gMVP", alphamissense = "AlphaMissense",
           esm2_score = "ESM-2 zero-shot")
COLS <- c(varity = PRED_COL[["VARITY"]], gmvp = PRED_COL[["gMVP"]],
          alphamissense = PRED_COL[["AlphaMissense"]], esm2_score = GOLD)
fct <- function(k) factor(NAMES[k], levels = NAMES[KEYS])
UNIT <- list(scale_x_continuous(limits = c(0, 1), breaks = c(0, 0.5, 1),
                                labels = c("0", "0.5", "1"), expand = c(0, 0)))

cal <- cal %>% filter(predictor %in% KEYS, n > 0) %>%
  mutate(pred = fct(predictor), col = COLS[predictor])
gap_poly <- bind_rows(lapply(KEYS, function(k) {
  d <- filter(cal, predictor == k) %>% arrange(mean_score)
  tibble(predictor = k, x = c(d$mean_score, rev(d$mean_score)),
         y = c(d$observed_rate, rev(d$mean_score)))
})) %>% mutate(pred = fct(predictor), col = COLS[predictor])
ece <- tibble(predictor = KEYS) %>%
  mutate(pred = fct(predictor),
         lab = sapply(KEYS, function(k) sprintf("ECE %.3f\nafter scaling %.3f",
                                                 cs[[k]]$ece, cs[[k]]$ece_after_temperature)))
pa <- ggplot() +
  annotate("segment", x = 0, y = 0, xend = 1, yend = 1, colour = INK2, linewidth = 0.35,
           linetype = "22") +
  geom_polygon(data = gap_poly, aes(x, y, fill = I(col)), alpha = 0.35) +
  geom_path(data = cal %>% arrange(mean_score), aes(mean_score, observed_rate, colour = I(col)),
            linewidth = 0.8) +
  geom_text(data = ece, aes(0.03, 0.97, label = lab), hjust = 0, vjust = 1, size = 2.2,
            lineheight = 1.0) +
  facet_wrap(~ pred, nrow = 1) +
  scale_x_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
  scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1")) +
  coord_equal(xlim = c(0, 1), ylim = c(0, 1)) +
  labs(x = "Mean score in bin", y = "Observed pathogenic rate") +
  theme_final() + theme(panel.spacing = unit(10, "pt"))

pb <- ggplot(cal) +
  geom_rect(aes(xmin = lo + 0.006, xmax = hi - 0.006, ymin = 0, ymax = n / 1000, fill = I(col))) +
  facet_wrap(~ pred, nrow = 1) + UNIT +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = "Score", y = "Thousand variants") +
  theme_final() + no_x_grid + theme(panel.spacing = unit(10, "pt"),
                                    strip.text = element_blank())

pu <- pu %>% filter(predictor %in% KEYS) %>%
  mutate(pred = fct(predictor), col = COLS[predictor], tail = auroc < 0.7)
br <- seq(0, 1, by = 0.05)
hist_d <- pu %>% mutate(b = pmin(floor(auroc / 0.05), 19)) %>% count(predictor, pred, b) %>%
  mutate(x0 = br[b + 1], x1 = br[b + 2], tail = x1 <= 0.7 + 1e-9,
         fill = ifelse(tail, INK, COLS[predictor]))
tail_lab <- pu %>% group_by(predictor, pred) %>%
  summarise(n = n(), below = mean(auroc < 0.7),
            worst = paste(head(gene_symbol[order(auroc)], 3), collapse = "\n"),
            .groups = "drop") %>%
  mutate(lab = sprintf("%s below 0.70\nof %d proteins", percent(below, accuracy = 0.1), n))
stopifnot(abs(tail_lab$below[tail_lab$predictor == "alphamissense"] -
              cs$alphamissense$protein_level$frac_proteins_below_0.7) < 1e-12)
ymax <- max(hist_d$n)
pc <- ggplot(hist_d) +
  annotate("rect", xmin = 0, xmax = 0.7, ymin = 0, ymax = Inf, fill = WASH) +
  geom_rect(aes(xmin = x0 + 0.004, xmax = x1 - 0.004, ymin = 0, ymax = n, fill = I(fill))) +
  geom_text(data = tail_lab, aes(0.03, ymax * 0.97, label = lab), hjust = 0, vjust = 1,
            size = 2.2, lineheight = 1.0) +
  geom_text(data = tail_lab, aes(0.03, ymax * 0.55, label = worst), hjust = 0, vjust = 1,
            size = 2.1, colour = INK2, lineheight = 1.0, fontface = "italic") +
  facet_wrap(~ pred, nrow = 1) + UNIT +
  scale_y_continuous(expand = expansion(mult = c(0, 0.05))) +
  labs(x = "AUROC within the protein", y = "Proteins") +
  theme_final() + no_x_grid + theme(panel.spacing = unit(10, "pt"),
                                    strip.text = element_blank())

fig <- tags(pa / pb / pc + plot_layout(heights = c(1.15, 0.6, 0.75)))
save_figure(fig, "S2_Fig", width = 7.5, height = 6.4)
