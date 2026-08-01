source("figures/theme_paper.R")

meta <- tryCatch(read_json_res("../../data/rebuild/build_metadata.json"),
                 error = function(e) NULL)
if (is.null(meta)) meta <- fromJSON("data/rebuild/build_metadata.json")
cn <- meta$counts

fun <- tibble::tibble(
  step = c("ClinVar rows (GRCh38)", "Unambiguous P/LP or B/LB",
           "Parsed missense", "Validated against UniProt"),
  n    = c(4476458, 1722790, 217219, cn$all)
) %>% mutate(step = factor(step, levels = rev(step)),
             frac = n / max(n))

pa <- ggplot(fun, aes(n, step)) +
  geom_col(fill = PAL$navy, alpha = 0.85, width = 0.62) +
  geom_text(aes(label = sprintf("%s   (%.0f%%)", comma(n), 100 * frac)),
            hjust = -0.08, size = 2.7, colour = PAL$ink, fontface = "bold") +
  scale_x_continuous(limits = c(0, 5.9e6), expand = c(0, 0)) +
  labs(x = NULL, y = NULL) +
  theme_paper() + grid_x_only +
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(colour = PAL$ink, size = 8.5),
        panel.grid.major.x = element_blank())

sets <- tibble::tibble(
  set = factor(c("MitoCarta 3.0", "Matched control"),
               levels = c("Matched control", "MitoCarta 3.0")),
  n   = c(cn$mito, cn$control),
  col = c(PAL$loses, PAL$navy)
)

pb <- ggplot(sets, aes(n, set)) +
  geom_col(aes(fill = I(col)), alpha = 0.85, width = 0.55) +
  geom_text(aes(label = sprintf("%s variants", comma(n))), hjust = -0.09,
            size = 2.8, colour = PAL$ink, fontface = "bold") +
  annotate("text", x = 0, y = 0.40, hjust = 0, vjust = 1, size = 2.6,
           colour = PAL$secondary, lineheight = 1.15,
           label = "955 genes each, matched on variants\nper gene and pathogenic fraction") +
  scale_x_continuous(limits = c(0, 17800), expand = c(0, 0)) +
  coord_cartesian(clip = "off", ylim = c(0.2, 2.6)) +
  labs(x = NULL, y = NULL) +
  theme_paper() + grid_x_only +
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(colour = PAL$ink, size = 8.5),
        panel.grid.major.x = element_blank())

sch <- expand.grid(x = 1:6, gene = c("gene A", "gene B")) %>%
  mutate(scheme = "Random 10-fold",
         held   = x %in% c(3, 6))
sch2 <- sch %>% mutate(scheme = "Leave-gene-out",
                       held = gene == "gene B")
sch3 <- sch %>% mutate(scheme = "In-sample", held = FALSE)
schemes <- bind_rows(sch3, sch, sch2) %>%
  mutate(scheme = factor(scheme,
                         levels = c("In-sample", "Random 10-fold",
                                    "Leave-gene-out")))

lab <- tibble::tibble(
  scheme = factor(c("In-sample", "Random 10-fold", "Leave-gene-out"),
                  levels = levels(schemes$scheme)),
  txt = c("null = 0.943", "null = 0.921", "null = 0.500"),
  col = c(PAL$null, PAL$null, PAL$loses)
)

pc <- ggplot(schemes, aes(x, gene)) +
  geom_point(aes(colour = I(ifelse(held, PAL$loses, PAL$navy)),
                 alpha = I(ifelse(held, 1, 0.30))), size = 3.4) +
  geom_text(data = lab, inherit.aes = FALSE,
            aes(x = 3.5, y = 2.85, label = txt, colour = I(col)),
            size = 2.9, fontface = "bold") +
  facet_wrap(~ scheme, nrow = 1) +
  scale_x_continuous(limits = c(0.4, 6.6)) +
  coord_cartesian(clip = "off", ylim = c(0.7, 3.15)) +
  labs(x = NULL, y = NULL,
       caption = "Filled = held out for testing.  Under leave-gene-out every test gene is unseen, so the null carries no information.") +
  theme_paper() +
  theme(panel.grid = element_blank(),
        axis.text.x = element_blank(),
        axis.text.y = element_text(colour = PAL$ink, size = 8),
        strip.text = element_text(colour = PAL$navy, size = 9,
                                  face = "bold", margin = margin(b = 5)),
        panel.spacing = unit(14, "pt"),
        plot.caption = element_text(colour = PAL$secondary, size = 7.5,
                                    hjust = 0, margin = margin(t = 6)),
        plot.caption.position = "plot")

fig <- (pa | pb) / pc +
  plot_layout(heights = c(1, 0.85)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 12, colour = PAL$navy))

save_figure(fig, "figure1_schematic", width = 7.8, height = 5.0)
