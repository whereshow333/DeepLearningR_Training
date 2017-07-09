https://analyticsdataexploration.com/deep-learning-using-h2o-in-r/

# Importing libraries
#### Let's import the data set from package MASS 
#### Also import h2o package for using h2o
library(MASS)
library(h2o)



# Reading the dataset
#### Storing the data set named "Boston" into DataFrame
DataFrame <- Boston

#### To get the Help on Boston dataset uncomment the following code 
#### help("Boston")

#### Lets have a look on Structure of Boston data 
str(DataFrame) 



# Data Exploration
#### Histogram of the target or outcome variable "medv"
hist(DataFrame$medv,col=colors()[100:110],
     breaks = 10,main="Histogram of medv",
     xlab="medv"
     )

####  Check the dimension of this data frame
dim(DataFrame)

####  Check first 3 rows
head(DataFrame,3)

#### Check the summary of each variable
summary(DataFrame)

#### This will give min and max value for each of the variable
apply(DataFrame,2,range)



# Data Transformation & H2o initialization
#### Seems like scale of each variable is not same

### Lets Normalize the data set variables in interval [0,1] 
### Normalization  is necessary so that each variable is scaled properly
### and none of the variables over dominates in the model 
### scale function will give min-max scaling here
### Below is the snippet of code for the same

maxValue <- apply(DataFrame, 2, max) 
minValue <- apply(DataFrame, 2, min)
DataFrame<-as.data.frame(scale(DataFrame,center = minValue,
                                         scale = maxValue-minValue))

										 
# H2o Initialization

####  Let's do H2o initialization.This will start H2o cluster in local machine 
####  There are options for running the same on servers
####  I'm using 2650 Megabytes of RAM out of 8GB RAM.You can choose according to 
####  your RAM configuration.
h2o.init(ip = "localhost",port = 54321,max_mem_size = "5G")



# Data Partition & Modelling
#### Lets partition the dataset into train and test data set

ind<-sample(1:nrow(DataFrame),400)
trainDF<-DataFrame[ind,]
testDF<-DataFrame[-ind,]


#### To know about the h2o.deeplearning function just run the follwing code 
#### by uncommenting it
#### ?h2o.deeplearning


#### Let's give a brief overview of parameters used in h2o.deeplearning function


#1. x is column names of predictor variable
#2. y is column name of target variable i.e medv
#3.activation=Tanh,TanhWithDropout,Rectifier,RectifierWithDropout
#              Maxout,etc                
#4.input_dropout_ratio=fraction of features for each 
#                      training row to be omitted in training(Its like random sampling for features)               


#5. l1,l2=regularization
#l1=makes weights 0
#l2=makes weights nearly zero not exactly zero

#6. loss="Automatic", "CrossEntropy" (for classification only),
#         "Quadratic", "Absolute" (experimental) or "Huber" 

#7. distribution=bernoulli,gaussian,multinomial,poisson,gamma,etc
#8. stopping metric="Auto",AUC,r2,logloss,etc
#9. stopping tolerance, metric-based stopping criterion
#10. nfolds =no. of folds for crossvalidation


#### Let's define x and y
y<-"medv"
x<-setdiff(colnames(DataFrame),y)

#### Fitting the Deeplearning  model in H2o
model<-h2o.deeplearning(x=x,
                        y=y,
                        seed = 1234,
                        training_frame = as.h2o(trainDF),
                        nfolds = 3,
                        stopping_rounds = 7,
                        epochs = 400,
                        overwrite_with_best_model = TRUE,
                        activation = "Tanh",
                        input_dropout_ratio = 0.1,
                        hidden = c(10,10),
                        l1 = 6e-4,
                        loss = "Automatic",
                        distribution = "AUTO",
                        stopping_metric = "MSE")

						
						
						
						
# Model Summary
#### Let's check the summary of this model
model



# Predictions on test data
predictions<-as.data.frame(predict(model,as.h2o(testDF)))
str(predictions)

#### MSE(Mean Squared Error)
sum((predictions$predict-testDF$medv)^2)/nrow(testDF)

#### plotting actual vs predicted values 
plot(testDF$medv,predictions$predict,col='blue',main='Real vs Predicted',pch=1,cex=0.9,type = "p",xlab = "Actual",ylab = "Predicted")
abline(0,1,col="black")



# Shutting down h2o cluster
#### Let's now shut down the H2o cluster.
h2o.shutdown(prompt=FALSE)
