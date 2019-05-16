
list.of.packages <- c("tensorflow","keras","purrr","dplyr", "plotly","dataPreparation")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(tensorflow)
library(keras)
library(dataPreparation)

library(readr)
library(knitr)
library(magrittr)
library(purrr)
library(ggplot2)
library(plotly)
library(dplyr)


wine.df <- read.csv('https://raw.githubusercontent.com/serivan/mldmlab/master/Datasets/Kaggle-Wine-train.csv', sep=",")
summary(wine.df)

wine.df$Quality <- as.factor(wine.df$Quality) 

constant_cols <- whichAreConstant(wine.df)
double_cols <- whichAreInDouble(wine.df)
bijections_cols <- whichAreBijection(wine.df)


row.has.na <- apply(wine.df,1, function(x){any(is.na(x))})
wine.df <- wine.df[!row.has.na,]
head(wine.df)

train<-wine.df %>% select(-Id, -Quality)
train_y<-to_categorical(as.matrix(wine.df %>% select(Quality)))

scales <- build_scales(dataSet = train, cols = colnames(data), verbose = TRUE)
print(scales)

train_x <- as.matrix(fastScale(dataSet = train, scales = scales, verbose = TRUE))
head(train_x)

set.seed(1)     


model = keras_model_sequential() 


model %>% 
  layer_dense(units = 20,  kernel_initializer = "uniform", activation = 'relu', input_shape = c(11)) %>% 
  layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 10, kernel_initializer = "uniform", activation = 'relu') %>%
#  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 2, kernel_initializer = "uniform", activation = 'sigmoid')

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(),
#    optimizer = "adam", 
  metrics = c('accuracy')
)

history <- model %>% fit(
  train_x,train_y,
  epochs = 200, 
 #batch_size = 250, 
  validation_split = 0.3
)

#Next, we evaluate the model and submit.

plot(history)

summary(model)

eval=model %>% evaluate(train_x, train_y)
eval

wine.test <- read.csv("https://raw.githubusercontent.com/serivan/mldmlab/master/Datasets/Kaggle-Wine-test.csv", stringsAsFactors = FALSE) 
#wine.test$Quality <- as.factor(wine.test$Quality) 
test<-wine.test %>% select(-Id)
test_x <- as.matrix(fastScale(dataSet = test, scales = scales, verbose = TRUE))


preds <-model %>% predict_classes(test_x) 
head(preds)

submission <- data.frame(Id = wine.test$Id,
                         Quality = preds)   
write.csv(submission, file = "../MySubmission.csv", row.names = FALSE) 

list.files(path = ".")
