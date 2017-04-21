library(vegan)
library(spaa)
library(ggcorrplot)
library(network)
library(sna)
library(ggplot2)
library(GGally)
library(reshape2)
library(colormap) #see https://github.com/bhaskarvk/colormap for colors
library(netassoc)
get_marker_name <- function(x){
  split_x <- strsplit(x, 'X')
  new_x <- split_x[[1]][2]
  # print(new_x)
  return(new_x)
  
}


plot_cor <- function(cormat, p.mat, plot_title){
  
  # cormat <- cor(raw_data)
  # p.mat <- cor_pmat(raw_data)
  # cormat[row(cormat)==col(cormat)] <- 0
  # cormat[upper.tri(cormat)] <- 0
  # p.mat[row(p.mat)==col(p.mat)] <- 1
  p.mat[upper.tri(p.mat)] <- 1
   
  melted_p <- melt(p.mat)
  colnames(melted_p)[which(colnames(melted_p)=='value')] <- 'pval'
  melted_cormat <- melt(cormat)
  melted_cormat <- merge(melted_cormat, melted_p, by=c('Var1', 'Var2'))
  melted_cormat$plot_text <- ifelse(melted_cormat$pval<0.05, round(melted_cormat$value, 4), "")
  
  p <- ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) + 
    geom_tile(color = "white") +
    scale_fill_gradient2(low = "blue", high = "red", mid = "white",
                         midpoint = 0, limit = c(-1,1), space = "Lab",
                         name="Pearson\nCorrelation") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                     size = 10, hjust = 1),
          axis.text.y = element_text(angle = 0, vjust = 1, 
                                     size = 10, hjust = 1))+
    coord_fixed() + 
    
    geom_text(aes(Var1, Var2, label = plot_text), color = "black", size = 1.2) +
    theme(
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      panel.grid.major = element_blank(), #element_line(color='grey', size=0.25), #element_blank(),
      panel.border = element_blank(),
      panel.background = element_blank(),
      axis.ticks = element_blank()) + #,
      # legend.justification = c(1, 0),
      # legend.position = c(0.5, 0.7),
      # legend.direction = "horizontal")+
    # guides(fill = guide_colorbar(barwidth = 10, barheight = 1,
                                 # title.position = "top", title.hjust = 0.5)) +
    ggtitle(plot_title)
  
  return(p)
  
}



calc_association_network <- function(subset_data, out_file_prefix, tissue_name){
  data_t <- t(subset_data)
  
  markers_in_mat <- rownames(data_t)
  n_markers_in_mat <- length(markers_in_mat)
  n_quadtrats <- ncol(data_t)
  pois_mat <- matrix(nrow = n_markers_in_mat, ncol = n_quadtrats)
  rownames(pois_mat)  <- rownames(data_t)
  colnames(pois_mat)  <- colnames(data_t)
  for(i in seq(1, length(markers_in_mat))){
    marker <- markers_in_mat[i]
    avg_marker_count <- mean(data_t[marker, ])
    
    random_dist <- rpois(avg_marker_count, n=n_quadtrats)
    pois_mat[marker, ] <- random_dist
    
    
  }
  n <- make_netassoc_network(data_t, nul = pois_mat,
                             method="partial_correlation",args=list(method="shrinkage"),
                             p.method= 'fdr' ,
                             numnulls=10, plot=F, alpha=0.05) 
  
  undir <- as.undirected(n$network_all)
  unfiltered <- n$matrix_spsp_obs
  unfiltered_mat <- matrix(unfiltered, nrow = nrow(unfiltered), ncol=ncol(unfiltered), dimnames = dimnames(unfiltered))

  unfiltered_p <- n$matrix_spsp_pvalue
  unfiltered_p_mat <- matrix(unfiltered_p, nrow = nrow(unfiltered_p), ncol=ncol(unfiltered_p), dimnames = dimnames(unfiltered_p))
  
  

  unfiltered_df <- setNames(melt(unfiltered_mat), c('Marker1', 'Marker2', 'Spatial_Correlation'))
  unfiltered_p_df <- setNames(melt(unfiltered_p_mat), c('Marker1', 'Marker2', 'significant'))
  sp_df <- merge(unfiltered_df, unfiltered_p_df, by=c('Marker1', 'Marker2'))
  
  graph_df <- as_data_frame(undir)
  print(graph_df)
  # unfiltered_df <- as_data_frame(undir_unfiltered)
  # colnames(graph_df)[which(colnames(graph_df)=='weight')] <- 'partial_correlation'
  # 
  # graph_file_out <- paste0(out_file_prefix, '_graphml.graphml')
  graph_df_out <- paste0(out_file_prefix, '_graph_df.csv')
  unfiltered_graph_df_out <- paste0(out_file_prefix, '_unfiltered.csv')
  # write_graph(undir, graph_file_out, format = 'graphml')
  write.csv(graph_df, graph_df_out, row.names = F)
  write.csv(sp_df, unfiltered_graph_df_out, row.names = F)
}

dst_dir <- './SpeciesAssociation/'
# data <- read.table('../Moffitt/Moffitt Ova TMA Pilot 3 Jan 26_Pano 01_D10.txt', header = T)
data <- read.csv('./quadrat_counts.csv')
non_markers <-   c('Quadrat')
all_colnames <- colnames(data)
markers <- all_colnames[! all_colnames %in% non_markers]

# markers_renamed <- lapply(markers, get_marker_name)
# markers_renamed <- unlist(markers_renamed)
# 
# print(markers_renamed)
# 
marker_df <- data[,markers]
calc_association_network(marker_df, '', '')
# print(marker_df)
# colnames(marker_df) <- markers_renamed
# corr <- cor(marker_df, method='pearson')
# p.mat <- cor_pmat(marker_df)

# cor_plot_title <- paste('Spatial Pearson Correlation Matrix')
# p <- plot_cor(corr, p.mat, cor_plot_title)
# ggsave('./corplot.png', p)

# cor_plot <-ggcorrplot(corr, p.mat = p.mat, hc.order = TRUE,
                      # type = "lower", insig = "blank", pch.cex = 20, lab=TRUE, show.legend = T, ggtheme=theme_minimal, title=cor_plot_title, method='square')