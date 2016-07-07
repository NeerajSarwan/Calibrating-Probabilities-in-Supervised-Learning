setwd("F:\\Internship_AV\\Datasets\\Blood_Donation")

list.files()

MultiLogLoss<-function(act, pred)
{
  eps = 1e-15;
  nr = length(pred)
  pred = matrix(sapply( pred, function(x) max(eps,x)), nrow = nr)      
  pred = matrix(sapply( pred, function(x) min(1-eps,x)), nrow = nr)
  ll = sum(act*log(pred) + (1-act)*log(1-pred))
  ll = ll * -1/(length(act))      
  return(ll);
}

train<-read.csv("train.csv")
test<-read.csv("test.csv")

train$Made.Donation.in.March.2007<-as.factor(train$Made.Donation.in.March.2007)

# removing the X column since it is irrelevant for our training and total colume column since it is perfectly correlated to number of donations

train<-train[-c(1,4)]

# splitting the train set into training and cv using random sampling
set.seed(221)
sub <- sample(nrow(train), floor(nrow(train) * 0.85))

training<-train[sub,]
cv<-train[-sub,]

# training a random forest model without any feature engineering or pre-processing

library(randomForest)
model_rf<-randomForest(Made.Donation.in.March.2007~.,data = training,keep.forest=TRUE,importance=TRUE)

result_cv<-as.data.frame(predict(model_rf,cv,type="prob"))

MultiLogLoss(as.numeric(as.character(cv$Made.Donation.in.March.2007)),result_cv$`1`)

# performing platt scaling on the dataset

dataframe<-data.frame(result_cv$`1`,cv$Made.Donation.in.March.2007)
colnames(dataframe)<-c("x","y")



model_log<-glm(y~x,data = dataframe,family = binomial)

result_cv_platt<-predict(model_log,dataframe[-2],type = "response")

MultiLogLoss(as.numeric(as.character(cv$Made.Donation.in.March.2007)),result_cv_platt)

# plotting reliability plots

reliability.plot <- function(obs, pred, bins=10, scale=T) {
  #  Plots a reliability chart and histogram of a set of predicitons from a classifier
  #
  # Args:
  #   obs: Vector of true labels. Should be binary (0 or 1)
  #   pred: Vector of predictions of each observation from the classifier. Should be real
  #       number
  #   bins: The number of bins to use in the reliability plot
  #   scale: Scale the pred to be between 0 and 1 before creating reliability plot
  require(plyr)
  library(Hmisc)
  
  min.pred <- min(pred)
  max.pred <- max(pred)
  min.max.diff <- max.pred - min.pred
  
  if (scale) {
    pred <- (pred - min.pred) / min.max.diff 
  }
  
  bin.pred <- cut(pred, bins)
  
  k <- ldply(levels(bin.pred), function(x) {
    idx <- x == bin.pred
    c(sum(obs[idx]) / length(obs[idx]), mean(pred[idx]))
  })
  
  is.nan.idx <- !is.nan(k$V2)
  k <- k[is.nan.idx,]  
  
  
  #subplot(hist(pred, xlab="", ylab="", main="", xlim=c(0,1), col="blue"), grconvertX(c(.8, 1), "npc"), grconvertY(c(0.08, .25), "npc"))
  return(k)
}


plot(c(0,1),c(0,1), col="grey",type="l",xlab = "Mean Prediction",ylab="Observed Fraction")
k<-reliability.plot(as.numeric(as.character(cv$Made.Donation.in.March.2007)),result_cv$`1`,bins = 5)
lines(k$V2, k$V1, xlim=c(0,1), ylim=c(0,1), xlab="Mean Prediction", ylab="Observed Fraction", col="red", type="o", main="Reliability Plot")
k<-reliability.plot(as.numeric(as.character(cv$Made.Donation.in.March.2007)),result_cv_platt,bins = 5)
lines(k$V2, k$V1, xlim=c(0,1), ylim=c(0,1), xlab="Mean Prediction", ylab="Observed Fraction", col="blue", type="o", main="Reliability Plot")


legend("topright",lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red"),legend = c("platt scaling","without plat scaling"))

test<-test[-c(1,4)]

result_test<-as.data.frame(predict(model_rf,newdata = test,type = "prob"))
head(result_test)

dataframe1<-data.frame(result_test$`1`)
colnames(dataframe1)<-c("x")
result_test_platt<-predict(model_log,dataframe1,type="response")

# fitting an isotonic regression now

fit.isoreg <- function(iso, x0) 
{
  o = iso$o
  if (is.null(o)) 
    o = 1:length(x)
  x = iso$x[o]
  y = iso$yf
  ind = cut(x0, breaks = x, labels = FALSE, include.lowest = TRUE)
  min.x <- min(x)
  max.x <- max(x)
  adjusted.knots <- iso$iKnots[c(1, which(iso$yf[iso$iKnots] > 0))]
  fits = sapply(seq(along = x0), function(i) {
    j = ind[i]
    
    # Handles the case where unseen data is outside range of the training data
    if (is.na(j)) {
      if (x0[i] > max.x) j <- length(x)
      else if (x0[i] < min.x) j <- 1
    }
    
    # Find the upper and lower parts of the step
    upper.step.n <- min(which(adjusted.knots > j))
    upper.step <- adjusted.knots[upper.step.n]
    lower.step <- ifelse(upper.step.n==1, 1, adjusted.knots[upper.step.n -1] )
    
    # Pefrom a liner interpolation between the start and end of the step
    denom <- x[upper.step] - x[lower.step] 
    denom <- ifelse(denom == 0, 1, denom)
    val <- y[lower.step] + (y[upper.step] - y[lower.step]) * (x0[i] - x[lower.step]) / (denom)
    
    # Ensure we bound the probabilities to [0, 1]
    val <- ifelse(val > 1, max.x, val)
    val <- ifelse(val < 0, min.x, val)
    val <- ifelse(is.na(val), max.x, val) # Bit of a hack, NA when at right extreme of distribution
    val
  })
  fits
}

# Remove any duplicates
idx <- duplicated(result_cv$`1`)
result_cv_unique <- result_cv$`1`[!idx]
cv$Made.Donation.in.March.2007<-as.numeric(as.character(cv$Made.Donation.in.March.2007))
cv_actual_unique<- cv$Made.Donation.in.March.2007[!idx]

iso.model <- isoreg(result_cv_unique, cv_actual_unique)

result_cv_isotonic <- fit.isoreg(iso.model, result_cv$`1`)

# plotting isotonic reliability plot
plot(c(0,1),c(0,1), col="grey",type="l",xlab = "Mean Prediction",ylab="Observed Fraction")
k<-reliability.plot(as.numeric(as.character(cv$Made.Donation.in.March.2007)),result_cv$`1`,bins = 5)
lines(k$V2, k$V1, xlim=c(0,1), ylim=c(0,1), xlab="Mean Prediction", ylab="Observed Fraction", col="red", type="o", main="Reliability Plot")
k<-reliability.plot(as.numeric(as.character(cv$Made.Donation.in.March.2007)),result_cv_isotonic,bins = 5)
lines(k$V2, k$V1, xlim=c(0,1), ylim=c(0,1), xlab="Mean Prediction", ylab="Observed Fraction", col="blue", type="o", main="Reliability Plot")

legend("topright",lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red"),legend = c("isotonic scaling","without isotonic scaling"))

MultiLogLoss(cv$Made.Donation.in.March.2007,result_cv$`1`)
MultiLogLoss(cv$Made.Donation.in.March.2007,result_cv_isotonic)

result_test_isotonic<-as.data.frame(fit.isoreg(iso.model,dataframe1$x))

write.csv(result_test_isotonic$`fit.isoreg(iso.model, dataframe1$x)`,"result.csv")
