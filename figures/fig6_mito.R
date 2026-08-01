source("figures/theme_paper.R")

ev  <- read_json_res("evaluation_report.json")
abl <- read_csv_res("ablation.csv")
pg  <- read_csv_res("proteingym_results.csv")

tag <- if ("2star" %in% names(ev$mito_vs_control)) "2star" else "0star"
mv  <- ev$mito_vs_control[[tag]]
keep <- c("varity", "alphamissense", "gmvp", "esm2_score",
          "gene_prior_insample", "gene_prior_randomsplit")
nice <- c(varity = "VARITY", alphamissense = "AlphaMissense", gmvp = "gMVP",
          esm2_score = "ESM-2 zero-shot",
          gene_prior_insample = "null (in-sample)",
          gene_prior_randomsplit = "null (random split)")

fa <- bind_rows(lapply(keep, function(k) {
  if (is.null(mv[[k]])) return(NULL)
  tibble::tibble(name = nice[[k]], delta = mv[[k]]$delta_auroc,
                 lo = mv[[k]]$ci[1], hi = mv[[k]]$ci[2], p = mv[[k]]$p)
})) %>%
  mutate(sig  = p < 0.05,
         col  = ifelse(sig, PAL$loses, PAL$navy),
         name = factor(name, levels = rev(unname(nice[keep]))))

pa <- ggplot(fa, aes(delta, name)) +
  annotate("rect", xmin = -0.01, xmax = 0.01, ymin = -Inf, ymax = Inf,
           fill = PAL$navy, alpha = 0.07) +
  geom_vline(xintercept = 0, colour = PAL$navy, linewidth = 0.6) +
  geom_linerange(aes(xmin = lo, xmax = hi, colour = I(col)), linewidth = 1) +
  geom_point(aes(colour = I(col), size = I(ifelse(sig, 3.2, 2.4)))) +
  geom_text(data = filter(fa, sig),
            aes(x = hi, label = sprintf("p = %.3f", p)),
            hjust = -0.18, size = 2.4, colour = PAL$loses, fontface = "bold") +
  annotate("text", x = 0, y = 0.28, hjust = 0.5, vjust = 1, size = 2.4,
           colour = PAL$secondary, label = "no difference") +
  annotate("text", x = -0.056, y = 6.85, hjust = 0, vjust = 0, size = 2.6,
           colour = PAL$loses, fontface = "bold",
           label = "mitochondrial worse") +
  scale_x_continuous(limits = c(-0.058, 0.040)) +
  coord_cartesian(clip = "off", ylim = c(0.5, 7.3)) +
  labs(x = "Mitochondrial minus matched control (AUROC)", y = NULL) +
  theme_paper() + grid_x_only +
  theme(axis.text.y = element_text(colour = PAL$ink, size = 8))

base <- abl$auroc[abl$ablation == "Full interpretable set"][1]
ab <- abl %>% filter(grepl("^minus ", ablation)) %>%
  mutate(group = sub("^minus ", "", ablation)) %>%
  left_join(abl %>% filter(grepl(" alone$", ablation)) %>%
              mutate(group = sub(" alone$", "", ablation)) %>%
              select(group, alone = auroc), by = "group") %>%
  mutate(group = fct_reorder(group, alone),
         useful = alone > 0.55,
         col = ifelse(useful, PAL$gains, PAL$loses))

pb <- ggplot(ab, aes(y = group)) +
  annotate("rect", xmin = 0.44, xmax = 0.55, ymin = -Inf, ymax = Inf,
           fill = PAL$loses, alpha = 0.07) +
  geom_vline(xintercept = 0.5, colour = PAL$navy, linewidth = 0.5,
             linetype = "dotted") +
  geom_segment(aes(x = 0.5, xend = alone, yend = group, colour = I(col)),
               linewidth = 2.6, lineend = "round", alpha = 0.30) +
  geom_point(aes(x = alone, colour = I(col)), size = 3.2) +
  geom_text(aes(x = alone, label = sprintf("%.3f", alone)), hjust = -0.32,
            size = 2.5, colour = PAL$ink, fontface = "bold") +
  annotate("text", x = 0.495, y = 6.8, hjust = 1, vjust = 0, size = 2.5,
           colour = PAL$loses, fontface = "bold",
           label = "at or below chance") +
  annotate("text", x = 0.5, y = 0.35, hjust = 0.5, vjust = 1, size = 2.5,
           colour = PAL$muted, label = "chance") +
  scale_x_continuous(limits = c(0.435, 0.75), breaks = seq(0.5, 0.7, 0.1)) +
  coord_cartesian(clip = "off", ylim = c(0.5, 7.2)) +
  labs(x = "AUROC using that feature group alone", y = NULL) +
  theme_paper() + grid_x_only +
  theme(axis.text.y = element_text(colour = PAL$ink, size = 8))

sp_cols <- grep("_spearman$", names(pg), value = TRUE)
n_cols  <- grep("_n$", names(pg), value = TRUE)

rho <- pg %>% select(gene, n_substitutions, all_of(sp_cols)) %>%
  pivot_longer(all_of(sp_cols), names_to = "method", values_to = "rho") %>%
  mutate(method = sub("_spearman$", "", method))
cov <- pg %>% select(gene, n_substitutions, all_of(n_cols)) %>%
  pivot_longer(all_of(n_cols), names_to = "method", values_to = "n_scored") %>%
  mutate(method = sub("_n$", "", method),
         frac = n_scored / n_substitutions) %>%
  select(gene, method, frac)

grid <- rho %>% left_join(cov, by = c("gene", "method")) %>%
  mutate(shown = !is.na(rho) & (is.na(frac) | frac >= 0.5),
         lab = ifelse(shown, sprintf("%.2f", rho), "—"),
         gene_lab = sprintf("%s\n%s subs", gene, comma(n_substitutions)),
         method = factor(method,
                         levels = c("AlphaMissense", "gMVP", "VARITY",
                                    "ESM-2 zero-shot")))

pc <- ggplot(grid, aes(gene_lab, fct_rev(method))) +
  geom_tile(aes(fill = ifelse(shown, rho, NA_real_)),
            colour = PAL$surface, linewidth = 2) +
  geom_text(aes(label = lab,
                colour = I(ifelse(shown & rho > 0.60, PAL$surface, PAL$ink))),
            size = 3.1, fontface = "bold") +
  scale_fill_gradient(low = "#DCF3F5", high = PAL$gains, na.value = "#ECEEF1",
                      guide = "none") +
  annotate("text", x = 3.62, y = 1, hjust = 0, size = 2.4,
           colour = PAL$muted, label = "— withheld:\n< 50% coverage") +
  coord_cartesian(clip = "off", xlim = c(0.5, 3.5)) +
  labs(x = NULL, y = NULL,
       caption = "Spearman ρ against experimental DMS fitness") +
  theme_paper() +
  theme(panel.grid.major = element_blank(),
        panel.background = element_rect(fill = PAL$surface, colour = NA),
        axis.text.x = element_text(colour = PAL$ink, size = 8.5,
                                   face = "bold", lineheight = 1.1),
        axis.text.y = element_text(colour = PAL$ink, size = 8.5),
        plot.caption = element_text(colour = PAL$secondary, size = 8,
                                    hjust = 0, margin = margin(t = 6)),
        plot.caption.position = "plot",
        plot.margin = margin(6, 66, 6, 6))

fig <- (pa | pb) / pc +
  plot_layout(heights = c(1, 0.78)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12, colour = PAL$navy))

save_figure(fig, "figure6_mitochondrial_case", width = 7.8, height = 6.0)
