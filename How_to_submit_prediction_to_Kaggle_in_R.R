list.of.packages <- c("rpart","rpart.plot")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(rpart)    
library(rpart.plot)



#Read in the training dataset
wine.df <- read.csv('https://raw.githubusercontent.com/serivan/mldmlab/master/Datasets/Kaggle-Wine-train.csv', sep=",")
summary(wine.df)

wine.df$Quality <- as.factor(wine.df$Quality)


#Build an rpart decision tree
set.seed(4786)     
simple.tree <- rpart(Quality ~ . - Id, data = wine.df)
prp(simple.tree)
wine.test <- read.csv('https://raw.githubusercontent.com/serivan/mldmlab/master/Datasets/Kaggle-Wine-test.csv', stringsAsFactors = FALSE)
preds <- predict(simple.tree, wine.test, type = "class") 
preds

submission <- data.frame(Id = wine.test$Id, Quality = preds)   
write.csv(submission, file = "MySubmission.csv", row.names = FALSE)
