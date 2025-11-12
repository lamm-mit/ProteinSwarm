library(ggseqlogo)
library(ggplot2)
library(dplyr)

csv_path <- "/example/csv_file.csv"
output_file <- "/example/output_file.png"

data <- read.csv(csv_path)

sequences <- data %>%
  filter(iteration >= 1 & iteration <= 16) %>%
  pull(sequence) %>%
  as.character()

custom_colors <- make_col_scheme(
  chars = c("A", "I", "L", "M", "F", "V", "P", "W",
            "R", "K",
            "D", "E",
            "S", "T", "N", "Q",
            "H", "Y",
            "C", "G"),
  groups = c("Hydrophobic", "Hydrophobic", "Hydrophobic", "Hydrophobic", "Hydrophobic", "Hydrophobic", "Hydrophobic", "Hydrophobic",
             "Positive", "Positive",
             "Negative", "Negative",
             "Polar", "Polar", "Polar", "Polar",
             "Aromatic", "Aromatic",
             "Special", "Special"),
  cols = c("#F7A09C", "#F9B29C", "#f19fa7", "#e38d96", "#eebc89", "#faa4ad", "#eebdc2", "#e39d6b",
           "#B7E1DC", "#B9EAE3",
           "#C6DBB9", "#C0E2AA",
           "#FBDF9C", "#E9E4AE", "#F9C99C", "#EFCEB4",
           "#F7A09C", "#EFB6B6",
           "#F8F7CC", "#F4EF9D")
)

p <- ggseqlogo(sequences, method = "bits", col_scheme = custom_colors) +
  labs(
    x = "Position",
    y = "Bits (Information Content)"
  ) +
  theme_minimal() +
  theme(
    axis.title = element_text(size = 16, face = "bold", color = "black"),
    axis.text = element_text(size = 16, face = "bold", color = "black"),
    axis.line = element_line(size = 1.2, color = "black"),
    axis.ticks = element_line(size = 1, color = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

print(p)

ggsave(output_file, plot = p, width = 10, height = 5, dpi = 600)
