library("CHRONOS")
library("doParallel")

#adjacencyMatrix2List = function(mat, keep.zero = FALSE) {
#    #if(is.null(rownames(mat))) rownames(mat) = as.character(seq_len(nrow(mat)))
#    #if(is.null(colnames(mat))) colnames(mat) = as.character(seq_len(ncol(mat)))
#    df = data.frame(from = rep(rownames(mat), times = ncol(mat)),
#        to = rep(colnames(mat), each = nrow(mat)),
#        value = as.vector(mat),
#        stringsAsFactors = FALSE)
#    if(!keep.zero) df = df[df$value != 0, , drop = FALSE]
#    return(df)
#}


# Function to convert adjacency matrix to adjacency list dataframe
adj_matrix_to_list <- function(adj_matrix) {
  from <- c()
  to <- c()
  weight <- c()
  adj_rownames = rownames(adj_matrix)
  adj_colnames = colnames(adj_matrix)
  
  for (i in 1:nrow(adj_matrix)) {
    edges <- which(adj_matrix[i, ] != 0)
    for (j in edges) {
      from <- c(from, adj_rownames[i])
      to <- c(to, adj_colnames[j])
      weight <- c(weight, adj_matrix[i, j])
    }
  }
  
  # Create dataframe
  adj_df <- data.frame(from = from, to = to, weight = weight)
  return(adj_df)
}


#cp KGML/*.xml /homes/onarykov/.CHRONOS/extdata/Downloads/KEGG/hsa/
pathways_list <- downloadKEGGPathwayList(org='hsa')
pathways_ids <- foreach(i = 1:length(pathways_list$Id)) %dopar% {substring(pathways_list$Id[i], 4, nchar(pathways_list$Id[i]))}
#print(pathways_ids)
#pathways <- downloadPathways(pathways_list$Id, org='hsa')
#test <- downloadPathways(c("04934", "01522"), org='hsa')
#print(test)
#pathways <- downloadPathways(pathways_ids, org='hsa')
#print(pathways)
#graphs <- createPathwayGraphs(org='hsa', pathways=pathways_ids)

for (i in 1:length(pathways_ids))
{
    graphs <- createPathwayGraphs(org='hsa', pathways=c(pathways_ids[[i]]))
    graph <- graphs$expanded[[1]]
    #print(nrows(graph))
    #print(ncols(graph))
    #print(rownames(graph))
    #print(graph)
    adj_list <- adj_matrix_to_list(graph) #adjacencyMatrix2List(graph)
    #print(adj_list)
    filename = paste(c('KEGG', 'KEGGID', paste(c('hsa', pathways_ids[[i]], '.csv'), collapse="")), collapse='/')
    print(filename)
    write.csv(adj_list,file=filename,row.names = F,quote=FALSE)

    #print(adj_list)

}

#print(graphs[1][1])
#y <- graphs$expanded$hsa04915
#
#hsa.pathways <- download_KEGG(species="hsa")
#term2gene <- hsa.pathways$KEGGPATHID2EXTID
