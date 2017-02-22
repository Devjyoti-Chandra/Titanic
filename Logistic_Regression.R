train=read.csv("train.csv",header = TRUE)
summary(train)
table(train$Survived)
summary(train$Age)
summary(train$SibSp)
summary(train$Parch)
summary(train$Fare)
index=!is.na(train$Age)
train1=subset(train, select = c("Survived","Pclass","Sex","Age","SibSp","Parch","Fare"))
train_age=train1[index,]
train_age=train_age[,-4]
test_age=train1[!index,]
test_age=test_age[,-4]
age=train1[,4]
a=age[index]
library(class)
age_na=knn(train_age,test_age, a , k = 3)
age_na=as.numeric(paste(age_na))
train$Age[!index]=age_na
write.csv(train,"train1.csv",row.names = FALSE)

mean_age=mean(train$Age)
sd_age=sd(train$Age)
train$Age_n=(train$Age-mean_age)/sd_age

mean_fare=mean(train$Fare)
sd_fare=sd(train$Fare)
train$Fare_n=(train$Fare-mean_fare)/sd_fare

model1=glm(Survived~Sex+Age_n+SibSp+Parch+Pclass+Fare_n,data=train,family = binomial)

model2=glm(Survived~Sex+Age_n+SibSp+Pclass+Fare_n,data=train,family = binomial)

model3=glm(Survived~Sex+Age_n+SibSp+Pclass,data=train,family = binomial)

test=read.csv("test.csv",header = TRUE)
str(test)
summary(test$Age)
summary(test$SibSp)
summary(test$Parch)
summary(test$Fare)
summary(test$Pclass)
table(test$Sex)
test$Fare[is.na(test$Fare)]=mean(train$Fare)

test1=subset(test, select = c("Pclass","Sex","Age","SibSp","Parch","Fare"))
index1=!is.na(test1$Age)
test_train_age=test1[index1,]
test_train_age=test_train_age[,-3]
test_test_age=test1[!index1,]
test_test_age=test_test_age[,-3]
age=test1[,3]
a=age[index1]
age_na=knn(test_train_age,test_test_age, a , k = 3)
age_na=as.numeric(paste(age_na))
test$Age[!index1]=age_na


test$Age_n=(test$Age-mean_age)/sd_age
test$Fare_n=(test$Fare-mean_fare)/sd_fare

predict=predict(model3,newdata=test,type="response")
predict=ifelse(predict>.6,1,0)
write.csv(predict,"output.csv")



train=train[,c(-4,-9,-11)]
dmy <- dummyVars(" ~ .", data = train)

train2 <- data.frame(predict(dmy, newdata = train))


trainPortion <- floor(nrow(train2))
trainSet <- train2[ 1:floor(trainPortion*.6),]
testSet <- train2[(floor(trainPortion*.6)+1):trainPortion,]

smallestError <- 100
for (depth in seq(1,10,1)) {
  for (rounds in seq(1,20,1)) {
    
    # train
    bst <- xgboost(data = as.matrix(trainSet[,-2]),
                   label = trainSet[,2],
                   max.depth=depth, nround=rounds,
                   objective = "reg:logistic", verbose=0)
    gc()
    
    # predict
    predictions <- predict(bst, as.matrix(testSet[,-2]), outputmargin=TRUE)
    err <- rmse(as.numeric(testSet[,2]), as.numeric(predictions))
    
    if (err < smallestError) {
      smallestError = err
      print(paste(depth,rounds,err))
    }     
  }
}  

cv <- 30
trainSet <- adultsTrsf[1:trainPortion,]
cvDivider <- floor(nrow(trainSet) / (cv+1))

smallestError <- 100
for (depth in seq(1,10,1)) { 
  for (rounds in seq(1,20,1)) {
    totalError <- c()
    indexCount <- 1
    for (cv in seq(1:cv)) {
      # assign chunk to data test
      dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
      dataTest <- trainSet[dataTestIndex,]
      # everything else to train
      dataTrain <- trainSet[-dataTestIndex,]
      
      bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                     label = dataTrain[,outcomeName],
                     max.depth=depth, nround=rounds,
                     objective = "reg:logistic", verbose=0)
      gc()
      predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
      
      err <- rmse(as.numeric(dataTest[,outcomeName]), as.numeric(predictions))
      totalError <- c(totalError, err)
    }
    if (mean(totalError) < smallestError) {
      smallestError = mean(totalError)
      print(paste(depth,rounds,smallestError))
    }  
  }
} 

###########################################################################
# Test both models out on full data set

trainSet <- adultsTrsf[ 1:trainPortion,]

# assign everything else to test
testSet <- adultsTrsf[(trainPortion+1):nrow(adultsTrsf),]

bst <- xgboost(data = as.matrix(trainSet[,predictors]),
               label = trainSet[,outcomeName],
               max.depth=4, nround=19, objective = "reg:linear", verbose=0)
pred <- predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE)
rmse(as.numeric(testSet[,outcomeName]), as.numeric(pred))

bst <- xgboost(data = as.matrix(trainSet[,predictors]),
               label = trainSet[,outcomeName],
               max.depth=3, nround=20, objective = "reg:linear", verbose=0)
pred <- predict(bst, as.matrix(testSet[,predictors]), outputmargin=TRUE)
rmse(as.numeric(testSet[,outcomeName]), as.numeric(pred))









bst <- xgboost(data = as.matrix(train2[,-2]),
               label = train2[,2],
               max.depth=5, nround=10,
               objective = "reg:logistic", verbose=0)
gc()

test=read.csv("test.csv",header = TRUE)
test=test[,c(-3,-8,-10)]
# predict
predictions <- predict(bst, as.matrix(test2), outputmargin=TRUE)

test$Pclass=as.factor(test$Pclass)
 test$Sex=as.factor(test$Sex)
 test$SibSp=as.factor(test$SibSp)
 test$Parch=as.factor(test$Parch)
test$Embarked=as.factor(test$Embarked)
train=train[,c(-4,-9,-11)]
dmy <- dummyVars(" ~ .", data = test)

test2 <- data.frame(predict(dmy, newdata = test))

