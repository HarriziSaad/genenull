suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(jsonlite)
  library(patchwork); library(scales); library(stringr); library(forcats)
})

RES <- "results"
FIG <- "figures/output"
dir.create(FIG, showWarnings = FALSE, recursive = TRUE)

PAL <- list(
  gains     = "#3FC1C9",
  loses     = "#FC5185",
  null      = "#8B5FBF",
  ink       = "#2B3E55",
  navy      = "#364F6B",
  secondary = "#5A6E86",
  muted     = "#8A9AAC",
  grid      = "#E2E6EA",
  panel     = "#F5F5F5",
  surface   = "#FFFFFF"
)

SUPERVISED <- c("VARITY_R_LOO", "gMVP", "VARITY")
pred_colour <- function(x) ifelse(x %in% SUPERVISED, PAL$loses, PAL$gains)

pretty_pred <- function(x) sub("_R_LOO$", "", x)

theme_paper <- function(base_size = 9.5) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      text             = element_text(colour = PAL$ink),
      plot.title       = element_blank(),
      plot.subtitle    = element_blank(),
      axis.title       = element_text(colour = PAL$secondary,
                                      size = base_size - 0.5),
      axis.text        = element_text(colour = PAL$muted, size = base_size - 1),
      axis.ticks       = element_blank(),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = PAL$grid, linewidth = 0.35),
      panel.background = element_rect(fill = PAL$panel, colour = NA),
      plot.background  = element_rect(fill = PAL$surface, colour = NA),
      legend.position  = "none",
      plot.margin      = margin(10, 10, 8, 8)
    )
}

grid_x_only <- theme(panel.grid.major.y = element_blank())
grid_y_only <- theme(panel.grid.major.x = element_blank())

read_json_res <- function(f) fromJSON(file.path(RES, f), simplifyVector = TRUE)
read_csv_res  <- function(f) read.csv(file.path(RES, f), check.names = FALSE)

save_figure <- function(plot, name, width, height) {
  ggsave(file.path(FIG, paste0(name, ".pdf")), plot,
         width = width, height = height, units = "in", device = cairo_pdf)
  ggsave(file.path(FIG, paste0(name, ".png")), plot,
         width = width, height = height, units = "in", dpi = 300,
         bg = PAL$surface)
  cat(sprintf("wrote %s  (%.1f x %.1f in)\n", name, width, height))
}
