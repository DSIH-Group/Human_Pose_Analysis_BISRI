install.packages("psych")
library(psych)

data <- read.csv("frameVectorDf.csv", row.names=1)
head(data)

#variance for one column was zero across all frames preventing parallel analysis thus we check which one that is
zero_var_cols <- names(data)[apply(data, 2, sd) == 0]
print(zero_var_cols) # <-- table feature 4 (front back ratio)

#we drop feature in order to facilitate parallel analysis <-- will also be dropped in frame vector 
data_clean <- data[ ,apply(data,2, sd) != 0 ]

result <- fa.parallel(data_clean, fa = "fa", n.iter = 100, main = "Parallel Analysis")

print(result) #suggest 5 factors

#varimax: makes each variable load highly onto as few factors as possible
#fm: how factors are computed 
  #fm = pa (principal axis factoring) more robust to non normal data
fa_results <- fa(data_clean, nfactors = 5, rotate = "varimax", fm="pa" )

#cutoff 0.4 to exclude weak loading's and make output easier to interpret
print(fa_results$loadings, cutoff=0.4)  #table feature 1 (active clinicians) loading is over 1 so we need to investigate that 

#exporting loadings
#write.csv(fa_results$loadings[], "efa_loadings.csv")


#attempting fa with 4 factors to see if we over extracted
fa_results4 <- fa(data_clean, nfactors = 4, rotate = "varimax", fm="pa" )
print(fa_results4$loadings, cutoff=0.4) #with this theirs no loading over 1, seems like it was an issue of overfactoring



#----removing table feature 1 and rerunning---- 
data_clean$Table_feature_1 <- NULL 
fa_results <- fa(data_clean, nfactors = 5, rotate = "varimax", fm="pa" )
print(fa_results$loadings, cutoff=0.4)   

fa_results <- fa(data_clean, nfactors = 4, rotate = "varimax", fm="pa" )
print(fa_results$loadings, cutoff=0.4)   

