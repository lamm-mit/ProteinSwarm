csv_path <- "sequence_features.csv"

library(tidyverse)
library(Rtsne)
library(ape)

options(warn = -1)
if (capabilities("cairo")) {
  options(bitmapType = "cairo")
}

theme_pub <- function() {
  theme_minimal() +
    theme(
      text = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      axis.text = element_text(size = 12),
      legend.title = element_text(size = 12, face = "bold"),
      legend.text = element_text(size = 11),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),
      legend.position = "right",
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    )
}

data <- read.csv(csv_path, stringsAsFactors = FALSE)

data$method <- case_when(
  grepl("^d.*_$", data$id) ~ "SCOPe",
  grepl("mpnn|MPNN", data$id, ignore.case = TRUE) ~ "ProteinMPNN",
  grepl("swarm|SWARM", data$id, ignore.case = TRUE) ~ "Swarm",
  TRUE ~ "SCOPe"
)

aa_cols <- grep("^aa_", colnames(data), value = TRUE)
additional_cols <- c("molecular_weight", "aromaticity")
existing_additional <- additional_cols[additional_cols %in% colnames(data)]

feature_cols <- c(aa_cols, existing_additional)
numerical_features <- data[, feature_cols]
numerical_features <- numerical_features %>% mutate_all(as.numeric)

numerical_complete <- complete.cases(numerical_features)
data_clean <- data[numerical_complete, ]

tsne_features_fair <- numerical_features[numerical_complete, ]

set.seed(42)
tsne_result_fair <- Rtsne(
  tsne_features_fair, 
  dims = 2, 
  perplexity = min(30, floor((nrow(tsne_features_fair) - 1) / 3)),
  verbose = FALSE,
  max_iter = 1000,
  check_duplicates = FALSE,
  pca = FALSE
)

tsne_df_fair <- data.frame(
  tSNE1 = tsne_result_fair$Y[, 1],
  tSNE2 = tsne_result_fair$Y[, 2],
  method = data_clean$method,
  id = data_clean$id,
  length = data_clean$length
)

method_colors <- c("SCOPe" = "#B40426", "ProteinMPNN" = "#F49A7B", "Swarm" = "#3B4CC0")
method_shapes <- c("SCOPe" = 20, "ProteinMPNN" = 20, "Swarm" = 20)

p1_fair <- suppressWarnings(
  ggplot(tsne_df_fair, aes(x = tSNE1, y = tSNE2)) +
    geom_point(color = "lightgray", alpha = 0.3, size = 0.8) +
    geom_point(aes(color = method, shape = method), 
               size = 2.5, alpha = 0.8, stroke = 0.5) +
    scale_color_manual(values = method_colors, name = "Method") +
    scale_shape_manual(values = method_shapes, name = "Method") +
    labs(
      title = "t-SNE Embedding: FAIR Comparison (Numerical Features Only)",
      subtitle = "Based on amino acid composition + physicochemical properties only",
      x = "t-SNE 1",
      y = "t-SNE 2"
    ) +
    theme_pub() +
    guides(
      color = guide_legend(override.aes = list(size = 4, alpha = 1)),
      shape = guide_legend(override.aes = list(size = 4, alpha = 1))
    )
)

ggsave("tsne_embedding.png", p1_fair, width = 5, height = 4, dpi = 300, bg = "white")

dist_matrix <- dist(tsne_features_fair, method = "euclidean")
nj_tree <- nj(dist_matrix)

if (!is.rooted(nj_tree)) {
  nj_tree <- root(nj_tree, outgroup = 1, resolve.root = TRUE)
}

nj_tree$tip.label <- data_clean$id
tip_methods <- data_clean$method[match(nj_tree$tip.label, data_clean$id)]
tip_colors <- method_colors[tip_methods]
names(tip_colors) <- nj_tree$tip.label

if (dev.cur() > 1) {
  dev.off()
}

png("phylogenetic_tree.png", width = 8, height = 6, units = "in", res = 300, bg = "white")
par(mar = c(5, 2, 4, 2))
plot(nj_tree, 
     type = "phylogram",
     show.tip.label = FALSE,
     main = "Phylogenetic Tree of Protein Sequences\n(Neighbor-Joining, Fair Comparison)",
     cex.main = 1.4,
     edge.width = 1.5)
tiplabels(pch = method_shapes[tip_methods], 
          col = tip_colors, 
          cex = 1.5,
          bg = "white")
legend("topright", 
       legend = names(method_colors),
       col = method_colors,
       pch = method_shapes[names(method_colors)],
       pt.cex = 2,
       cex = 1.2,
       bg = "white",
       box.lwd = 1.5,
       title = "Method")
dev.off()
