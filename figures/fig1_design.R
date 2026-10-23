source("figures/theme_final.R")
suppressPackageStartupMessages({ library(ggraph); library(tidygraph) })

meta <- fromJSON("data/rebuild/build_metadata.json")
cn <- meta$counts
fx <- read_json_res("figure_data_extra.json")
psq <- fromJSON(file.path(FDATA, "pair_square.json"))
gb <- read_fdata("pair_blocks.csv") %>% mutate(n = p + b, frac = p / n)
stopifnot(sum(gb$n) == cn$all, nrow(gb) == psq$n_genes)

FRAC_COL <- c(BENIGN, "#FFB36B", "#F2545B", PATH)
nodes <- bind_rows(tibble(name = "root", n = 0, frac = NA_real_),
                   gb %>% transmute(name = gene_symbol, n, frac))
gr <- tbl_graph(nodes = nodes, edges = tibble(from = 1L, to = 2:nrow(nodes)))
set.seed(1)
lay <- create_layout(gr, layout = "circlepack", weight = n)
leaves <- as_tibble(lay) %>% filter(leaf)
stopifnot(nrow(leaves) == nrow(gb))
top <- leaves %>% filter(name %in% c("FBN1", "KMT2D", "DNAH11", "SCN1A", "BRCA2", "LDLR")) %>%
  mutate(tc = ifelse(frac > 0.55, SURF, INK))
pa <- ggplot() +
  ggforce::geom_circle(data = leaves, aes(x0 = x, y0 = y, r = r, fill = frac),
                       colour = SURF, linewidth = 0.06, n = 28) +
  geom_text(data = top, aes(x, y, label = name, colour = I(tc)), size = 1.45,
            fontface = "bold") +
  scale_fill_gradientn(colours = FRAC_COL, values = c(0, 0.35, 0.65, 1), limits = c(0, 1),
                       breaks = c(0, 0.5, 1), labels = c("all benign", "half", "all pathogenic"),
                       name = NULL,
                       guide = guide_colourbar(barheight = unit(4, "pt"),
                                               barwidth = unit(120, "pt"),
                                               ticks.colour = SURF)) +
  coord_equal(clip = "off") +
  theme_void(base_family = FAMILY) +
  theme(legend.position = "bottom", legend.text = element_text(size = 6.4, colour = INK2),
        legend.margin = margin(2, 0, 0, 0), plot.margin = margin(6, 6, 4, 6))

N <- c(4476458, 1722790, 217219, cn$all)
stopifnot(psq$n_pathogenic + psq$n_benign == N[4])
A1 <- (N[1] - N[2]) / N[1]
H2 <- N[3] / N[2]
C3 <- (N[3] - N[4]) / N[3] * (1 - A1)
PW <- psq$n_pathogenic / N[4] * (1 - A1 - C3)
blk <- tribble(
  ~xmin,          ~xmax,           ~ymin, ~ymax, ~fill,
  0,              A1,              0,     1,     "#DCD7E8",
  A1,             1,               H2,    1,     "#BEB8CF",
  A1,             A1 + C3,         0,     H2,    "#7E7894",
  A1 + C3,        A1 + C3 + PW,    0,     H2,    PATH,
  A1 + C3 + PW,   1,               0,     H2,    BENIGN) %>%
  mutate(area = (xmax - xmin) * (ymax - ymin))
stopifnot(abs(blk$area - c(N[1] - N[2], N[2] - N[3], N[3] - N[4], psq$n_pathogenic,
                           psq$n_benign) / N[1]) < 1e-12)
pb <- ggplot() +
  geom_rect(data = blk, aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax, fill = I(fill)),
            colour = SURF, linewidth = 0.7) +
  annotate("text", x = A1 / 2, y = 0.56, size = 3.4, fontface = "bold", colour = INK,
           label = comma(N[1] - N[2])) +
  annotate("text", x = A1 / 2, y = 0.50, size = 2.15, colour = INK2, vjust = 1,
           lineheight = 0.95, label = "not unambiguously\npathogenic or benign") +
  annotate("text", x = (A1 + 1) / 2, y = 0.62, size = 3.4, fontface = "bold", colour = INK,
           label = comma(N[2] - N[3])) +
  annotate("text", x = (A1 + 1) / 2, y = 0.56, size = 2.15, colour = INK2, vjust = 1,
           lineheight = 0.95, label = "not missense, or\nprotein change\nnot parsed") +
  annotate("text", x = 0, y = 1.035, hjust = 0, vjust = 0, size = 3.4, fontface = "bold",
           colour = INK, label = comma(N[1])) +
  annotate("text", x = 0.335, y = 1.035, hjust = 0, vjust = 0, size = 2.15, colour = INK2,
           label = "ClinVar rows\non GRCh38", lineheight = 0.95) +
  annotate("segment", x = 1.03, xend = 1.03, y = 0.005, yend = H2 - 0.005, colour = INK2,
           linewidth = 0.4) +
  annotate("text", x = 1.06, y = 0.19, hjust = 0, vjust = 0, size = 3.0,
           fontface = "bold", colour = INK, label = comma(N[4])) +
  annotate("text", x = 1.06, y = 0.18, hjust = 0, vjust = 1, size = 2.1,
           colour = INK2, label = sprintf("analysed, in %s genes", comma(psq$n_genes))) +
  annotate("richtext", x = 1.06, y = 0.085, hjust = 0, vjust = 1, size = 2.1, colour = INK,
           fill = NA, label.colour = NA, label.padding = unit(0, "pt"), family = FAMILY,
           lineheight = 1.25,
           label = sprintf(paste0("<span style='color:%s'>\u25a0</span> %s pathogenic<br>",
                                  "<span style='color:%s'>\u25a0</span> %s benign"),
                           PATH, comma(psq$n_pathogenic), BENIGN,
                           comma(psq$n_benign))) +
  annotate("segment", x = A1 + C3 / 2, xend = A1 + C3 / 2, y = -0.01, yend = -0.06,
           colour = INK2, linewidth = 0.3) +
  annotate("text", x = A1 + C3 / 2 - 0.01, y = -0.075, hjust = 1, vjust = 1, size = 2.05,
           colour = INK2, lineheight = 0.95,
           label = sprintf("%s not matching\nthe UniProt sequence", comma(N[3] - N[4]))) +
  annotate("text", x = 1.06, y = 0.62, hjust = 0, size = 2.05, colour = MUTED,
           lineheight = 0.95, label = "area is the\nnumber of records") +
  scale_x_continuous(expand = c(0, 0)) +
  scale_y_continuous(expand = c(0, 0)) +
  coord_equal(xlim = c(-0.01, 1.62), ylim = c(-0.2, 1.17), clip = "off") +
  theme_void(base_family = FAMILY) + theme(plot.margin = margin(10, 8, 4, 10))

lev <- c("In-sample", "Random 10-fold", "Leave-gene-out")
hs <- read_fdata("null_score_hist.csv") %>%
  mutate(scheme = factor(scheme, levels = lev), h = ifelse(label == 1, share, -share),
         fill = ifelse(label == 1, PATH, BENIGN))
vals <- tibble(scheme = factor(lev, levels = lev),
               v = c(fx$null_all_variants$in_sample, fx$null_all_variants$random_split, 0.5))
side <- tibble(scheme = factor(lev[1], levels = lev), y = c(0.93, -0.93),
               lab = c("pathogenic variants", "benign variants"), vj = c(1, 0))
pc <- ggplot(hs) +
  geom_rect(aes(xmin = lo + 0.005, xmax = hi - 0.005, ymin = 0, ymax = h, fill = I(fill))) +
  geom_hline(yintercept = 0, colour = INK2, linewidth = 0.3) +
  geom_text(data = vals, aes(0.98, 0.93, label = sprintf("null AUROC %.3f", v)), hjust = 1,
            vjust = 1, size = 2.8, fontface = "bold", colour = INK) +
  geom_text(data = side, aes(0.02, y, label = lab, vjust = vj), hjust = 0, size = 2.15,
            colour = INK2) +
  facet_wrap(~ scheme, nrow = 1) +
  scale_x_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"), expand = c(0, 0)) +
  scale_y_continuous(breaks = c(-1, -0.5, 0, 0.5, 1), labels = c("100%", "50%", "0", "50%", "100%"),
                     expand = c(0, 0)) +
  coord_cartesian(xlim = c(0, 1), ylim = c(-1.02, 1.02)) +
  labs(x = "Score the null gives the variant (its gene's pathogenic fraction)",
       y = "Share of class") +
  theme_final() + no_x_grid +
  theme(panel.spacing = unit(14, "pt"),
        strip.text = element_text(face = "bold", size = 8, colour = INK, hjust = 0,
                                  margin = margin(t = 4, b = 3)))

fig <- ((pa | pb) + plot_layout(widths = c(1.05, 1))) / pc +
  plot_layout(heights = c(1.5, 1)) +
  plot_annotation(tag_levels = "a") &
  theme(plot.tag = element_text(face = "bold", size = 11, colour = INK, family = FAMILY))
save_figure(fig, "Fig1", width = 7.5, height = 7.2)
