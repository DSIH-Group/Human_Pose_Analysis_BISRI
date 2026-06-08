##running efa across all frames

data <- read.csv("completeFrameMatrix.csv", row.names=1)
head(data)

result <- fa.parallel(data, fa = "fa", n.iter = 100, main = "Parallel Analysis")
print(result) #suggest 3 factors


fa_results <- fa(data, nfactors = 3, rotate = "varimax", fm="pa" )
print(fa_results$loadings, cutoff=0.4)   

#----removing table feature 1 and rerunning---- 
data$Table_feature_1 <- NULL 
fa_results <- fa(data, nfactors = 3, rotate = "varimax", fm="pa" )
print(fa_results$loadings, cutoff=0.4)   