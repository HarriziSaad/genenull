suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(jsonlite)
  library(patchwork); library(scales); library(forcats); library(ggrepel)
  library(ggtext)
})

RES <- "results"
FIG <- "figures/output"
FDATA <- file.path(FIG, "data")
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

INK    <- "#16121F"
INK2   <- "#4A4560"
MUTED  <- "#8E89A0"
FAINT  <- "#CDC8DA"
GRID   <- "#ECE9F2"
WASH   <- "#F6F4FA"
WASH2  <- "#ECE8F4"
SURF   <- "#FFFFFF"

NULL_COL  <- "#4A1FD6"
NULL_DARK <- "#26107E"
NULL_T    <- "#B9A9F7"
CLASS_COL <- c("Clinically trained" = "#FF1F7A", "Population-tuned" = "#00B4D8",
               "Population-free" = "#FFA400", "Not listed" = "#A29DB2")
ORANGE   <- CLASS_COL[["Clinically trained"]]
ORANGE_T <- "#FF9CC4"
AQUA     <- CLASS_COL[["Population-tuned"]]
GOLD     <- CLASS_COL[["Population-free"]]
PRED_COL <- c(AlphaMissense = AQUA, VARITY = ORANGE, gMVP = ORANGE_T)
CONTRAST <- "#3B3254"
JADE_RAMP <- c("#E2F7FC", "#A3E7F6", "#45CBE8", "#0096C2", "#005A82")
PATH     <- "#C1003E"
BENIGN   <- "#F9C77E"
NEG_GREY <- "#A9A4B9"
null_gradient <- function() grid::linearGradient(c(NULL_COL, NULL_DARK), x1 = 0, y1 = 1,
                                                 x2 = 0, y2 = 0)

FAMILY <- "Arial"
update_geom_defaults("text", list(family = FAMILY, colour = INK))
update_geom_defaults("label", list(family = FAMILY))
update_geom_defaults(ggrepel::GeomTextRepel, list(family = FAMILY))
update_geom_defaults(ggtext::GeomRichText, list(family = FAMILY))

theme_final <- function(base = 8) {
  theme_minimal(base_size = base, base_family = FAMILY) +
    theme(
      text               = element_text(colour = INK, family = FAMILY),
      plot.title         = element_blank(),
      plot.subtitle      = element_blank(),
      axis.title         = element_text(size = base - 0.5, colour = INK2),
      axis.text          = element_text(size = base - 0.8, colour = INK2),
      axis.ticks         = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major   = element_line(colour = GRID, linewidth = 0.35),
      panel.background   = element_rect(fill = SURF, colour = NA),
      plot.background    = element_rect(fill = SURF, colour = NA),
      legend.position    = "none",
      legend.title       = element_text(size = base - 0.8, colour = INK2, face = "bold"),
      legend.text        = element_text(size = base - 0.8, colour = INK),
      legend.key.size    = unit(8, "pt"),
      legend.margin      = margin(0, 0, 0, 0),
      legend.box.margin  = margin(0, 0, 2, 0),
      strip.text         = element_text(face = "bold", size = base - 0.3, colour = INK, hjust = 0),
      plot.margin        = margin(8, 8, 6, 6)
    )
}

no_y_grid <- theme(panel.grid.major.y = element_blank())
no_x_grid <- theme(panel.grid.major.x = element_blank())
no_grid   <- theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())
top_left_legend <- theme(legend.position = "top", legend.justification = "left",
                         legend.direction = "horizontal",
                         legend.box.margin = margin(0, 0, 2, -4))

fill_key <- function(values, title = NULL, nrow = 1) {
  list(scale_fill_identity(guide = guide_legend(title = title, nrow = nrow,
                                                override.aes = list(colour = NA)),
                           breaks = unname(values), labels = names(values)))
}

tags <- function(p) {
  p + plot_annotation(tag_levels = "a") &
    theme(plot.tag = element_text(face = "bold", size = 11, colour = INK, family = FAMILY))
}

read_json_res <- function(f) {
  txt <- paste(readLines(file.path(RES, f), warn = FALSE), collapse = "\n")
  fromJSON(gsub("\\bNaN\\b", "null", txt), simplifyVector = TRUE)
}
read_csv_res <- function(f) read.csv(file.path(RES, f), check.names = FALSE)
read_fdata <- function(f) read.csv(file.path(FDATA, f), check.names = FALSE)

minus <- function(s) gsub("-", "\u2212", s)
fmt_signed <- function(x, d = 3) minus(sprintf(paste0("%+.", d, "f"), x))
fmt_p_md <- function(p) {
  if (p == 0) return("p < 0.001")
  if (p < 1e-4) {
    e <- floor(log10(p)); m <- round(p / 10^e)
    if (m == 10) { m <- 1; e <- e + 1 }
    return(sprintf("p = %d \u00d7 10<sup>%s</sup>", m, minus(as.character(e))))
  }
  if (p < 0.001) return(sprintf("p = %.4f", p))
  if (p < 0.1) return(sprintf("p = %.3f", p))
  sprintf("p = %.2f", p)
}

pretty_pred <- function(x) {
  x <- sub("_score$", "", x)
  x <- sub("^Polyphen2_HVAR$", "PolyPhen-2", x)
  x <- sub("^VARITY_R_LOO$", "VARITY", x)
  x <- sub("^MisFit_S$", "MisFit", x)
  x
}
atlas_class <- function(x) dplyr::recode(x, "Clinical-trained" = "Clinically trained",
                                         "not in Atlas" = "Not listed")

save_figure <- function(plot, name, width, height) {
  stopifnot(width <= 7.5, height <= 8.75)
  ggsave(file.path(FIG, paste0(name, ".pdf")), plot, width = width, height = height,
         units = "in", device = cairo_pdf)
  ggsave(file.path(FIG, paste0(name, ".png")), plot, width = width, height = height,
         units = "in", dpi = 300, bg = SURF)
  ragg::agg_tiff(file.path(FIG, paste0(name, ".tif")), width = width, height = height,
                 units = "in", res = 600, compression = "lzw", background = SURF)
  print(plot)
  invisible(dev.off())
  cat(sprintf("wrote %s (%.2f x %.2f in)\n", name, width, height))
}
