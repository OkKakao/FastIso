args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 4) {
  stop("usage: envipat_peaks.R FORMULA THRESHOLD REL_TO ALGO")
}

formula <- args[[1]]
threshold <- as.numeric(args[[2]])
rel_to <- as.numeric(args[[3]])
algo <- as.integer(args[[4]])

user_lib <- file.path(
  Sys.getenv("USERPROFILE"),
  "R",
  "win-library",
  paste(R.version$major, strsplit(R.version$minor, "\\.")[[1]][1], sep = ".")
)
if (dir.exists(user_lib)) {
  .libPaths(c(user_lib, .libPaths()))
}

suppressPackageStartupMessages(library(enviPat))
data(isotopes, package = "enviPat")

pattern <- isopattern(
  isotopes,
  formula,
  threshold = threshold,
  charge = FALSE,
  algo = algo,
  rel_to = rel_to,
  verbose = FALSE
)[[1]]

result <- data.frame(
  mass = as.numeric(pattern[, 1]),
  prob = as.numeric(pattern[, 2])
)
write.csv(result, row.names = FALSE)
