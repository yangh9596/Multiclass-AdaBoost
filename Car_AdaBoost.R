# INPUT DATA --------------------------------------------------------------
setwd("/Users/yangh/Desktop/yangh/Courses/PKU/Machine Learning/HW2")
dat <- read.csv("car_re.csv",sep = ",", header = TRUE)
for(i in 1:7){
  dat[,i] <- as.factor(dat[,i])
}
# reconstructure
index <- sample(nrow(dat),size = 2*nrow(dat)/3)
train <- dat[index,]
test <- dat[-index,]

# Naive Baysian --------------------------------------------------------------
# Naive Baysian with Laplace Smoothing
NaiveBaysian <- function(Data, ColLabel, Distr, Lamda){
  Label <- as.numeric(levels(Data[,ColLabel]))
  K <- length(Label)
  M <- ncol(Data)-1
  
  AttrMat <- list(NULL)
  for(j in 1:M){
    AttrMat[[j]] <- as.numeric(levels(Data[,j]))
  }
  
  VoteMatrix <- matrix(NA, nrow = nrow(Data), ncol = K)
  Prediction <- vector("numeric", length = nrow(Data))
  temp <- vector("numeric", length = M)
  
  # Train
  ProdY <- vector("numeric", length = K)
  ProdX_Y <- AttrMat

  for(k in 1:K){
    
    # calculate prob of P(Y=yi)
    ProdY[k] <- (sum(Distr*(Data[,ColLabel]==Label[k]))+Lamda)/(sum(Distr)+K*Lamda)
    for(j in 1:M){
      for(l in seq(1,length(AttrMat[[j]]))){
        # calculate P(X = xj|Y)
        ProdX_Y[[j]][l] <- (sum(Distr*((Data[,j]==AttrMat[[j]][l])&
                                        (Data[,ColLabel]==Label[k])))+Lamda)/
          (sum(Distr*(Data[,ColLabel]==Label[k]))+Lamda*length( AttrMat[[j]] ))
      }
    }
    
    # Predict
    for(i in 1:nrow(Data)){
      for(m in 1:M){
        index <- which(AttrMat[[m]]==Data[i,m])
        temp[m] <- ProdX_Y[[m]][index]
      }
      VoteMatrix[i,k] <- ProdY[k]*prod(temp)
    }
  }
  
  
  
  for(i in 1:nrow(Data)){
    Prediction[i] <- which.max(VoteMatrix[i,])
  }
  
  rm(temp)
  
  List <- list(Prediction = Prediction, Vote = VoteMatrix)
  return(Prediction)
}

# AdaBoost --------------------------------------------------------------
###########################################################################
# Data: a data frame                                                      #
# ColLabel: an integer, indicating the column index of the label (or Y)   #
# MaxInter: to set the maximumn of iterations (or # of weak learners)     #
# OUTPUT: a list, including Alphas, weak learner parameters,              #
#         predictions and the vote matrix                                 #
###########################################################################

MultiAdaBoost <- function(Data, ColLabel, MaxIter, Lamda){
  # Y = {1,2,3,...,k}
  Label <- as.numeric(levels(as.factor(Data[,ColLabel])))
  k <- length(Label)
  # Weights = {a1,a2,...,at} (t = MaxIter)
  Weights <- vector(mode = "numeric")
  # hypothetic models = {h1, h2,..., ht} (t = MaxIter)
  Model <- list(NULL)
  Pred <- list(NULL)
  VoteMatrix <- matrix(data = NA, nrow = nrow(Data), ncol = k)
  # Weak learner parameters
  eta <- vector(mode = "numeric", length = MaxIter)
  M <- vector(mode = "numeric", length = MaxIter)
  
  
  # Initialize --------------------------------------------------------------
  Distr <- rep(1/(nrow(Data)*1), nrow(Data)) 
  Alpha <- 0
  
  for(i in seq(1,MaxIter)){
    
    # Train Models ------------------------------------------------------------
    ########################################################
    ######### Please Insert Weak Learner Code Here #########
    # native bayesian
   
    # We need a training function format like this
    # Model[length[Model]+1] <- model(Data, Distr)
    
    # gives out predictions
    # Lamda = 0.5
    Pred[[i]] <- NaiveBaysian(Data, 7, Distr, Lamda)
    
    # Store Weak Learner Parameters
    
    ########################################################
    
    # Choose Alpha ------------------------------------------------------------
    error <- sum(Distr*(Pred[[i]]!= Data[,ColLabel]))/sum(Distr)
    Alpha <- 0.5*log((1-error)/error)
    
    # Update Alpha vector
    Weights[length(Weights) + 1] <- Alpha
    
    # Update Distribution -----------------------------------------------------
    Distr <- Distr*exp(-Alpha*(Pred[[i]] == Data[,ColLabel]))
    Z <- sum(Distr)
    Distr <- Distr/Z
  }
    
  # Output Final Hypo ------------------------------------------------------------
  for(i in seq(1:k)){
    for(j in seq(1:nrow(Data))){
      VoteMatrix[j,i] <- 0
      for(t in seq(1:MaxIter)){
        VoteMatrix[j,i] <- VoteMatrix[j,i] + Weights[t]*(Pred[[t]][j] == i)
      }
    }
  }
  Forecast <- apply(VoteMatrix, MARGIN = 1, FUN = which.max)
  List <- list(Alphas = Weights, Forecast = Forecast, Votes = VoteMatrix, 
               Model = Model)
  return(List)
}

###################################################
# test error
# Naive Baysian
Test.NB <- function(Data,test, ColLabel, Distr, Lamda){
  Label <- as.numeric(levels(Data[,ColLabel]))
  K <- length(Label)
  M <- ncol(Data)-1
  
  AttrMat <- list(NULL)
  for(j in 1:M){
    AttrMat[[j]] <- as.numeric(levels(Data[,j]))
  }
  
  VoteMatrix <- matrix(NA, nrow = nrow(test), ncol = K)
  Prediction <- vector("numeric", length = nrow(test))
  temp <- vector("numeric", length = M)
  
  # Train
  ProdY <- vector("numeric", length = K)
  ProdX_Y <- AttrMat
  
  for(k in 1:K){
    
    # calculate prob of P(Y=yi)
    ProdY[k] <- (sum(Distr*(Data[,ColLabel]==Label[k]))+Lamda)/(sum(Distr)+K*Lamda)
    for(j in 1:M){
      for(l in seq(1,length(AttrMat[[j]]))){
        # calculate P(X = xj|Y)
        ProdX_Y[[j]][l] <- (sum(Distr*((Data[,j]==AttrMat[[j]][l])&
                                         (Data[,ColLabel]==Label[k])))+Lamda)/
          (sum(Distr*(Data[,ColLabel]==Label[k]))+Lamda*length( AttrMat[[j]] ))
      }
    }
    
    # Predict
    for(i in 1:nrow(test)){
      for(m in 1:M){
        index <- which(AttrMat[[m]]==test[i,m])
        temp[m] <- ProdX_Y[[m]][index]
      }
      VoteMatrix[i,k] <- ProdY[k]*prod(temp)
    }
  }
  rm(temp)
  
  for(i in 1:nrow(test)){
    Prediction[i] <- which.max(VoteMatrix[i,])
  }
  
  
  List <- list(Prediction = Prediction, Vote = VoteMatrix)
  return(Prediction)
}

############################################################################
# Test AdaBoost
Test.AB <- function(Data, test, ColLabel, MaxIter, Lamda){
  # Y = {1,2,3,...,k}
  Label <- as.numeric(levels(as.factor(Data[,ColLabel])))
  k <- length(Label)
  # Weights = {a1,a2,...,at} (t = MaxIter)
  Weights <- vector(mode = "numeric")
  # hypothetic models = {h1, h2,..., ht} (t = MaxIter)
  Model <- list(NULL)
  Pred <- list(NULL)
  Test.Pred <- list(NULL)
  VoteMatrix <- matrix(data = NA, nrow = nrow(test), ncol = k)

  
  # Initialize --------------------------------------------------------------
  Distr <- rep(1/(nrow(Data)*1), nrow(Data)) 
  Alpha <- 0
  
  for(i in seq(1,MaxIter)){
    
    # Train Models ------------------------------------------------------------
    ########################################################
    ######### Please Insert Weak Learner Code Here #########
    # native bayesian
    
    # We need a training function format like this
    # Model[length[Model]+1] <- model(Data, Distr)
    
    # gives out predictions
    # Lamda = 0.5
    Pred[[i]] <- NaiveBaysian(Data, 7, Distr, Lamda)
    Test.Pred[[i]] <- Test.NB(Data, test,7, Distr, Lamda)
    # Store Weak Learner Parameters
    
    ########################################################
    
    # Choose Alpha ------------------------------------------------------------
    error <- sum(Distr*(Pred[[i]]!= Data[,ColLabel]))/sum(Distr)
    Alpha <- 0.5*log((1-error)/error)
    
    # Update Alpha vector
    Weights[length(Weights) + 1] <- Alpha
    
    # Update Distribution -----------------------------------------------------
    Distr <- Distr*exp(-Alpha*(Pred[[i]] == Data[,ColLabel]))
    Z <- sum(Distr)
    Distr <- Distr/Z
  }
  
  # Output Final Hypo ------------------------------------------------------------
  
  
  for(i in seq(1:k)){
    for(j in seq(1:nrow(test))){
      VoteMatrix[j,i] <- 0
      for(t in seq(1:MaxIter)){
        VoteMatrix[j,i] <- VoteMatrix[j,i] + Weights[t]*(Test.Pred[[t]][j] == i)
      }
    }
  }
  Forecast <- apply(VoteMatrix, MARGIN = 1, FUN = which.max)
  List <- list(Alphas = Weights, Forecast = Forecast, Votes = VoteMatrix, 
               Model = Model)
  return(List)
}


############################################################################
# Code example -------------------------------------------------------------
# max iteration = 5 (at most 5 weak learners)
car.adaboost <- MultiAdaBoost(train, 7, 6, Lamda = 0)

table(car.adaboost$Forecast,train$X1.4)


# get weights of weak learners
car.adaboost$Alphas

# get forecasting results
car.adaboost$Forecast

# get the vote matrix
car.adaboost$Votes

# error rate
error.NB <- vector("numeric", length = 21)
for(i in seq(0,20)){
  car.NB <- NaiveBaysian(train, 7, Distr, (i/10))
  error.NB[i+1] <- 1-sum(car.NB==train[,7])/nrow(train)
}

plot(x=seq(0,20)/10,y=error.NB,type = "l", xlab ="Lamda", ylab = "Train Error",col = "blue", lwd = 2)
car.NB <- NaiveBaysian(train, 7, Distr, 0)
table(car.NB,train$X1.4)


error.AB <- vector("numeric", length = 10)
for(i in seq(1,10)){
  train.adaboost <-  MultiAdaBoost(train, 7, i, Lamda = 0)
  error.AB[i] <- 1-sum(train.adaboost$Forecast==train[,7])/nrow(train)
}

plot(x= seq(1,10),y=error.AB,type = "l", xlab ="Number of Weak Learners",
     ylab = "Train Error",col = "blue", lwd = 2)




# Naive Bayesian
Distr <- rep(1/nrow(train),nrow(train))
car.NBtest <- Test.NB(train,test,7,Distr, 0)
error.test.NB <- vector("numeric", length = 21)
for(i in seq(0,20)){
  car.test.NB <- Test.NB(train,test, 7, Distr, (i/10))
  error.test.NB[i+1] <- 1-sum(car.test.NB==test[,7])/nrow(test)
}

plot(x=seq(0,20)/10,y=error.test.NB,type = "l", xlab ="Lamda", ylab = "Test Error",col = "blue", lwd = 2)


# AdaBoost
error.test.AB <- vector("numeric", length = 10)
for(i in seq(1,10)){
  test.adaboost <-  Test.AB(train,test, 7, i, Lamda = 0)
  error.test.AB[i] <- 1-sum(test.adaboost$Forecast==test[,7])/nrow(test)
}

plot(x= seq(1,10),y=error.test.AB,type = "l", xlab ="Number of Weak Learners",
     ylab = "Test Error",col = "blue", lwd = 2)

