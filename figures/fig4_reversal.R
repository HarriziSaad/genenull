source("figures/theme_final.R")
suppressPackageStartupMessages(library(ggbeeswarm))

ss  <- read_json_res("same_set_comparison.json")
rep <- read_json_res("replication_matched.json")
wg  <- read_json_res("within_gene_ranking_modern.json")
pg  <- read_csv_res("within_gene_ranking_modern.csv")
dms <- read_csv_res("dms_arbitration.csv")
dr  <- read_json_res("dms_robustness.json")
bt  <- read.csv(gzfile(file.path(RES, "same_set_bootstrap.csv.gz")), check.names = FALSE)
rb  <- read.csv(gzfile(file.path(RES, "replication_bootstrap.csv.gz")), check.names = FALSE)

NAME <- c(alphamissense = "AlphaMissense", varity = "VARITY_R_LOO", gmvp = "gMVP")
NICE <- c(alphamissense = "AlphaMissense", varity = "VARITY", gmvp = "gMVP")
PAIRS <- tibble(win = c("alphamissense", "alphamissense", "varity"),
                lose = c("varity", "gmvp", "gmvp"),
                key = c("varity|alphamissense", "alphamissense|gmvp", "varity|gmvp"),
                sgn = c(-1, 1, 1))

hdr <- function(x, y, prob = 0.95) {
  lx <- range(x) + c(-1, 1) * diff(range(x)) * 0.3
  ly <- range(y) + c(-1, 1) * diff(range(y)) * 0.3
  k <- MASS::kde2d(x, y, n = 150, lims = c(lx, ly))
  dz <- k$z[cbind(findInterval(x, k$x), findInterval(y, k$y))]
  cl <- contourLines(k$x, k$y, k$z, levels = quantile(dz, 1 - prob))
  bind_rows(lapply(seq_along(cl), function(i) tibble(x = cl[[i]]$x, y = cl[[i]]$y, piece = i)))
}
plane_data <- function(boot, comps, cname) {
  draws <- bind_rows(lapply(seq_len(nrow(PAIRS)), function(i) {
    r <- PAIRS[i, ]
    tibble(pair = i, win = r$win, lose = r$lose,
           x = r$sgn * boot[[paste0(r$key, "|global")]],
           y = r$sgn * boot[[paste0(r$key, "|pooled")]])
  }))
  obs <- bind_rows(lapply(seq_len(nrow(PAIRS)), function(i) {
    r <- PAIRS[i, ]
    a <- strsplit(r$key, "\\|")[[1]]
    lab <- paste(NAME[a[1]], "-", NAME[a[2]])
    g <- comps[comps$comparison == lab & comps$metric == "global", ]
    p <- comps[comps$comparison == lab & comps$metric == "pooled", ]
    tibble(pair = i, win = r$win, lose = r$lose, x = r$sgn * g$diff, y = r$sgn * p$diff,
           p_global = g$p, p_within = p$p, lo = r$sgn * g[[cname]])
  }))
  cont <- draws %>% group_by(pair, win, lose) %>%
    group_modify(~ hdr(.x$x, .x$y)) %>% ungroup()
  list(draws = draws, obs = obs, cont = cont)
}
ci_of <- function(comps) {
  comps$lo <- if ("ci_low" %in% names(comps)) comps$ci_low else sapply(comps$ci, `[`, 1)
  comps
}
A <- plane_data(bt, ci_of(as_tibble(ss$comparisons)), "lo")
cr <- rep$all_variants$pairwise
cr <- tibble(comparison = cr$comparison, metric = cr$metric, diff = cr$diff,
             p = cr$p, lo = sapply(cr$ci, `[`, 1))
C <- plane_data(rb, cr, "lo")
for (P in list(A, C)) for (i in seq_len(nrow(PAIRS))) {
  dx <- P$draws$x[P$draws$pair == i]
  lo_pub <- P$obs$lo[i]
  q <- if (PAIRS$sgn[i] > 0) quantile(dx, 0.025, type = 7) else -quantile(-dx, 0.025, type = 7)
  stopifnot(abs(unname(q) - lo_pub) < 1e-12)
}

XL <- c(-0.0175, 0.0125); YL <- c(-0.0045, 0.0225)
plane <- function(P, xlab, ylab, labpos) {
  lab <- P$obs %>% mutate(lx = labpos$x, ly = labpos$y, hj = labpos$h, vj = labpos$v,
                          txt = sprintf("%s over %s", NICE[win], NICE[lose]))
  ggplot() +
    annotate("rect", xmin = XL[1], xmax = 0, ymin = 0, ymax = YL[2], fill = WASH) +
    annotate("text", x = XL[1] + 0.0006, y = YL[2] - 0.0006, hjust = 0, vjust = 1, size = 2.3,
             colour = INK, fontface = "bold", label = "reversed") +
    annotate("text", x = XL[1] + 0.0006, y = YL[2] - 0.0021, hjust = 0, vjust = 1, size = 2.1,
             colour = INK2, lineheight = 0.95, label = "wins within genes,\nloses globally") +
    annotate("text", x = XL[2] - 0.0006, y = YL[2] - 0.0006, hjust = 1, vjust = 1, size = 2.3,
             colour = INK, fontface = "bold", label = "same winner") +
    geom_hline(yintercept = 0, colour = INK2, linewidth = 0.3) +
    geom_vline(xintercept = 0, colour = INK2, linewidth = 0.3) +
    geom_point(data = P$draws, aes(x, y, colour = I(PRED_COL[NICE[win]])), size = 0.45,
               alpha = 0.30, stroke = 0) +
    geom_path(data = P$cont, aes(x, y, group = interaction(pair, piece),
                                 colour = I(PRED_COL[NICE[lose]])), linewidth = 0.6) +
    geom_point(data = P$obs, aes(x, y, fill = I(PRED_COL[NICE[win]])), shape = 21,
               colour = SURF, size = 2.3, stroke = 0.7) +
    geom_segment(data = lab, aes(x = x, y = y, xend = lx, yend = ly), colour = INK2,
                 linewidth = 0.25) +
    geom_text(data = lab, aes(lx, ly, label = txt, hjust = hj, vjust = vj), size = 2.2,
              colour = INK) +
    scale_x_continuous(labels = function(v) minus(sprintf("%+.3f", v)),
                       breaks = seq(-0.015, 0.01, 0.005)) +
    scale_y_continuous(labels = function(v) minus(sprintf("%+.3f", v)),
                       breaks = seq(0, 0.02, 0.005)) +
    coord_equal(xlim = XL, ylim = YL, expand = FALSE, clip = "off") +
    labs(x = xlab, y = ylab) +
    theme_final() + no_grid +
    theme(panel.border = element_rect(fill = NA, colour = FAINT, linewidth = 0.4))
}
LABPOS <- list(x = c(-0.0168, -0.0035, 0.0118), y = c(0.0022, 0.0205, -0.0028),
               h = c(0, 0.5, 1), v = c(1.35, -0.4, 1.35))
pa <- plane(A, "Difference in global AUROC", "Difference in pooled within-gene AUROC", LABPOS)
pc <- plane(C, "Difference in global AUROC", "Difference in pooled within-protein AUROC",
            LABPOS)

XS <- c(-0.40, 0.40)
swarm <- function(rows, xlab, unit_word, size) {
  pts <- bind_rows(lapply(seq_len(nrow(rows)), function(i) {
    r <- rows[i, ]
    tibble(row = r$row, d = r$d[[1]]) %>%
      mutate(col = ifelse(d > 0, PRED_COL[r$first], ifelse(d < 0, PRED_COL[r$second], MUTED)),
             dd = pmin(pmax(d, XS[1]), XS[2]))
  }))
  ggplot(pts) +
    geom_segment(data = rows, aes(x = 0, xend = 0, y = row - 0.42, yend = row + 0.42),
                 colour = INK, linewidth = 0.35) +
    geom_quasirandom(aes(dd, row, colour = I(col)), orientation = "y", width = 0.40,
                     size = size, stroke = 0, alpha = 0.9, method = "quasirandom") +
    geom_richtext(data = rows, aes(XS[2] + 0.03, row,
                                   label = sprintf("<b>%s better</b><br>%s of %s %s<br>%s",
                                                   first, comma(wins), comma(n), unit_word,
                                                   ptxt)),
                  hjust = 0, vjust = 0.5, size = 2.15, colour = INK, fill = NA,
                  label.colour = NA, label.padding = unit(0, "pt"), lineheight = 1.05) +
    geom_richtext(data = rows, aes(XS[1] - 0.03, row,
                                   label = sprintf("<b>%s better</b><br>%s%s", second,
                                                   comma(losses), ttxt)),
                  hjust = 1, vjust = 0.5, size = 2.15, colour = INK2, fill = NA,
                  label.colour = NA, label.padding = unit(0, "pt"), lineheight = 1.05) +
    scale_x_continuous(breaks = seq(-0.4, 0.4, 0.2), labels = function(v)
      minus(sprintf("%+.1f", v))) +
    scale_y_reverse(breaks = NULL) +
    coord_cartesian(xlim = c(-0.78, 0.86), ylim = c(3.5, 0.5), expand = FALSE, clip = "off") +
    labs(x = xlab, y = NULL) +
    theme_final() + no_y_grid +
    theme(panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3))
}
pair_row <- function(first, second, wins, losses, n, p, d, row, ties = TRUE) {
  ttxt <- if (ties) sprintf("<br>%s ties", comma(sum(d == 0))) else ""
  tibble(first = first, second = second, wins = wins, losses = losses, n = n, p = p,
         d = list(d), row = row, ptxt = fmt_p_md(p), ttxt = ttxt)
}
W <- wg$wilcoxon
gb <- bind_rows(
  pair_row("AlphaMissense", "VARITY",
           W$`VARITY_R_LOO vs AlphaMissense`$non_ties - W$`VARITY_R_LOO vs AlphaMissense`$wins,
           W$`VARITY_R_LOO vs AlphaMissense`$wins, W$`VARITY_R_LOO vs AlphaMissense`$non_ties,
           W$`VARITY_R_LOO vs AlphaMissense`$p, pg$alphamissense - pg$varity, 1),
  pair_row("AlphaMissense", "gMVP", W$`AlphaMissense vs gMVP`$wins,
           W$`AlphaMissense vs gMVP`$non_ties - W$`AlphaMissense vs gMVP`$wins,
           W$`AlphaMissense vs gMVP`$non_ties, W$`AlphaMissense vs gMVP`$p,
           pg$alphamissense - pg$gmvp, 2),
  pair_row("VARITY", "gMVP", W$`VARITY_R_LOO vs gMVP`$wins,
           W$`VARITY_R_LOO vs gMVP`$non_ties - W$`VARITY_R_LOO vs gMVP`$wins,
           W$`VARITY_R_LOO vs gMVP`$non_ties, W$`VARITY_R_LOO vs gMVP`$p,
           pg$varity - pg$gmvp, 3))
stopifnot(all(mapply(function(x, w) sum(x > 0) == w, gb$d, gb$wins)),
          all(mapply(function(x, l) sum(x < 0) == l, gb$d, gb$losses)))
pb <- swarm(gb, "Difference in within-gene AUROC, per gene", "genes", 0.42)

PA <- dr$per_assay
db <- bind_rows(
  pair_row("AlphaMissense", "VARITY", PA$`AlphaMissense vs VARITY_R_LOO`$wins,
           PA$`AlphaMissense vs VARITY_R_LOO`$n - PA$`AlphaMissense vs VARITY_R_LOO`$wins,
           PA$`AlphaMissense vs VARITY_R_LOO`$n, PA$`AlphaMissense vs VARITY_R_LOO`$p,
           dms$AlphaMissense_rho - dms$VARITY_R_LOO_rho, 1, ties = FALSE),
  pair_row("AlphaMissense", "gMVP", PA$`AlphaMissense vs gMVP`$wins,
           PA$`AlphaMissense vs gMVP`$n - PA$`AlphaMissense vs gMVP`$wins,
           PA$`AlphaMissense vs gMVP`$n, PA$`AlphaMissense vs gMVP`$p,
           dms$AlphaMissense_rho - dms$gMVP_rho, 2, ties = FALSE),
  pair_row("VARITY", "gMVP", PA$`VARITY_R_LOO vs gMVP`$wins,
           PA$`VARITY_R_LOO vs gMVP`$n - PA$`VARITY_R_LOO vs gMVP`$wins,
           PA$`VARITY_R_LOO vs gMVP`$n, PA$`VARITY_R_LOO vs gMVP`$p,
           dms$VARITY_R_LOO_rho - dms$gMVP_rho, 3, ties = FALSE))
stopifnot(all(mapply(function(x, w) sum(x > 0) == w, db$d, db$wins)))
pd <- swarm(db, "Difference in Spearman \u03c1 with experiment, per assay", "assays", 1.5)

fig <- tags((pa | pb) / (pc | pd) + plot_layout(widths = c(1, 1.35)))
save_figure(fig, "Fig4", width = 7.5, height = 6.4)
