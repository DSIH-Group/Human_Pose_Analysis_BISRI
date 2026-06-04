install.packages("psych")
library(psych)

data <- read.csv("frameVectorDf.csv", row.names=1)
print(data)

#variance for one column was zero across all frames preventing parallel analysis thus we check which one that is
zero_var_cols <- names(data)[apply(data, 2, sd) == 0]
print(zero_var_cols) # <-- table feature 4 (front back ratio)

#we drop feature in order to facilitate parallel analysis 
data_clean <- data[ ,apply(data,2, sd) != 0 ]

result <- fa.parallel(data_clean, fa = "fa", n.iter = 100, main = "Parallel Analysis")

print(result) #suggest 5 factors

#varimax: makes each variable load highly onto as few factors as possible
#fm: how factors are computed 
  #fm = pa (principal axis factoring) more robust to non normal data
fa_results <- fa(data_clean, nfactors = 5, rotate = "varimax", fm="pa" )

#cutoff 0.3 to exclude weak loading's and make output easier to interpret
print(fa_results$loadings, cutoff=0.4) 

#table feature 1 (active clinicians) loading is over 1 so we need to investigate that 
print(fa_results)

