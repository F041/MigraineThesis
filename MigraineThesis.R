library(forecast)
library(ggplot2)
library(lsr)
library(SCCS)
library(magrittr)
library(gnm)
require(survival)
library(glarma)
library(caret)
library(zoo)
library(gam)
library(lme4)
library(Hmisc)
options("scipen"=999)

### Caricamento dati ------
dati<-read.table("C:/Users/F041/Downloads/MigraineOctober2019 - Senza Inquinamento.csv", header = TRUE, sep = ",", dec=",")
dati<-dati[1:999,]
dati$Date=as.Date(dati$Date,format = "%d/%m/%Y");dati$Date
dati$Mese<- as.yearmon(dati$Date, format = "%d/%m/%Y")
dati$Mese<-as.numeric(format(dati$Mese,"%m"))
dati$Quarti<- as.yearqtr(dati$Date, format = "%d/%m/%Y")
dati$Date=as.numeric(dati$Date)

### Modello inferenziale escluse collinearità -----
glm1<-glm(Migraine~Uni.Lesson+Volo+FS+Quarti+Gym...P.A.+
            AutoAltaPercorrenza+exp(Stress),data=dati)
summary(glm1)
exp(glm1$coefficients)
sprintf("%.4f", exp(glm1$coefficients))


#### Preparazione dati ed Oversampling ------
library(ROSE)
prova<-ovun.sample(Migraine ~ ., data=dati, method="both", N=1600, seed=40); table(prova$data$Migraine)
over<-prova$data
split <- createDataPartition(y=over$Migraine, p = 0.66, list = FALSE)
train <- over[split,]
test <- over[-split,]
levels(train$Migraine) <- c("Yes", "No")


####  Lasso -----
set.seed(1234)
grid = expand.grid(.alpha=1,.lambda=seq(0, 1, by = 0.01))
Control=trainControl(method= "cv",number=10, classProbs=TRUE)
glm_lasso=train(Migraine~Uni.Lesson+Volo+FS+Quarti+Gym...P.A.+
                  AutoAltaPercorrenza+Mese+Chocolate+exp(Stress), method = "glmnet", data=train,
                trControl = Control, tuneLength=5, tuneGrid=grid, metric="Spec")
glm_lasso
plot(glm_lasso)
getTrainPerf(glm_lasso)

### Albero ------
set.seed(1)
levels(train$Migraine) <- c("Yes", "No")
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
rpartTuneCvA <- train(Migraine ~ Uni.Lesson+Volo+FS+Quarti+Gym...P.A.+
                        AutoAltaPercorrenza+Mese+Chocolate+exp(Stress), data = train, method = "rpart",
                      tuneLength = 10, na.action=na.exclude,
                      trControl = cvCtrl, metric="Spec")

rpartTuneCvA
getTrainPerf(rpartTuneCvA)
plot(varImp(object=rpartTuneCvA),main="train tuned - Variable Importance")


### Random Forest -----
set.seed(1)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
rfTune <- train(Migraine ~ Uni.Lesson+Volo+FS+Quarti+Gym...P.A.+
                  AutoAltaPercorrenza+Mese+Chocolate+exp(Stress), data = train, method = "rf",
                tuneLength = 10,
                trControl = cvCtrl, metric="Spec")
rfTune
plot(varImp(object=rfTune),main="train tuned - Variable Importance")

### Neural Net -----
set.seed(2)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
NNTune <- train(Migraine ~ Uni.Lesson+Volo+FS+Quarti+Gym...P.A.+
                  AutoAltaPercorrenza+Mese+Chocolate+Stress, data = train, method = "nnet",
                tuneLength = 8,
                trControl = cvCtrl, metric="metric", preProcess="range") # Non andare oltre il 8 di tuneLength 
NNTune
plot(varImp(object=NNTune),main="train tuned - Variable Importance")

### Xtreme Gradient -----
set.seed(3)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
XGBTune <- train(Migraine ~ Uni.Lesson+Volo+FS+Quarti+Gym...P.A.+
                  AutoAltaPercorrenza+Mese+Chocolate+exp(Stress), data = train, method = "xgbTree",
                tuneLength = 6,
                trControl = cvCtrl, metric="metric") # Non andare oltre il 6 di tuneLength 
XGBTune
plot(varImp(object=XGBTune),main="train tuned - Variable Importance")

### Regole indotte ------
set.seed(5)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
PRIMTune <- train(Migraine ~ Uni.Lesson+Volo+FS+Quarti+Gym...P.A.+
                        AutoAltaPercorrenza+Mese+Chocolate+exp(Stress), data = train, method = "rpart",
                      tuneLength = 10, na.action=na.exclude,
                      trControl = cvCtrl, metric="Spec")

PRIMTune
getTrainPerf(PRIMTune)
plot(varImp(object=PRIMTune),main="train tuned - Variable Importance")

### Adaboost ------
set.seed(5)
cvCtrl <- trainControl(method = "cv", number=10, search="grid", classProbs = TRUE,
                       summaryFunction = twoClassSummary)
AdaBTune <- train(Migraine ~ Uni.Lesson+Volo+FS+Quarti+Gym...P.A.+
                    AutoAltaPercorrenza+Mese+Chocolate+Stress, data = train, method = "adaboost",
                  tuneLength = 6, na.action=na.exclude,
                  trControl = cvCtrl, metric="Spec")

AdaBTune
getTrainPerf(AdaBTune)
plot(varImp(object=AdaBTune),main="train tuned - Variable Importance")

### Risultati ----
results <- resamples(list(Tree=rpartTuneCvA, RandomForest=rfTune, 
                          NeuralNet=NNTune, XGBoosting=XGBTune,
                          PatientRules=PRIMTune, AdaBoost=AdaBTune));results #Se metto il lasso non funziona
bwplot(results)
# test difference of accuracy using bonferroni adjustement
Diffs <- diff(results)
summary(Diffs)


p_prova = predict(rfTune      , test, "prob")
head(p_prova)

# estimate probs P(M)
test$p1 = predict(glm_lasso       , test, "prob")[,2]
test$p2 = predict(rpartTuneCvA, test, "prob")[,2]
test$p3 = predict(rfTune    , test, "prob")[,2]
test$p4 = predict(NNTune     , test, "prob")[,2]
test$p5 = predict(XGBTune, test, "prob")[,2]
test$p6 = predict(AdaBTune, test, "prob")[,2]

# roc values
r1=roc(Migraine ~ p1, data = test)
r2=roc(Migraine ~ p2, data = test)
r3=roc(Migraine ~ p3, data = test)
r4=roc(Migraine ~ p4, data = test)
r5=roc(Migraine ~ p5, data = test)
r6=roc(Migraine ~ p6, data = test)

plot(r1) #lasso nero
plot(r2,add=T,col="red") #Tree
plot(r3,add=T,col="blue") #RF
plot(r4,add=T,col="yellow") #NN
plot(r5,add=T,col="violet") #XGB
plot(r6,add=T,col="orange") #Ada
legend("bottomright", c("Tree", "Random Forest", "NN","XGB","ADA"))
text.col=c("red","blue","yellow","violet","orange")


### Scelta soglia migliore modello -----
library(pROC)
ROCit_obj <- rocit(score=test$p3,class=test$Migraine)
plot(ROCit_obj) #suggerisce un 1-0.1, quindi 0.9

#qualcosa di più approfondito
pROC_obj <- roc(test$Migraine,test$p3,
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)

plot(pROC_obj, print.thres = quantile(pROC_obj$thresholds, seq(0.4,0.95,0.4))) #suggerisce un 0.895, quindi uno 0.9


### Risultati miglior modello, RF ------
test$labelsp3<-as.factor(ifelse(test$p3<0.9,0,1))
test$Migraine<-as.factor(test$Migraine)
cm<-confusionMatrix(test$labelsp3, test$Migraine, positive="1") #Tante metriche, bellissimo

draw_confusion_matrix <- function(cm) {
  
  total <- sum(cm$table)
  res <- as.numeric(cm$table)
  
  # Generate color gradients. Palettes come from RColorBrewer.
  greenPalette <- c("#F7FCF5","#E5F5E0","#C7E9C0","#A1D99B","#74C476","#41AB5D","#238B45","#006D2C","#00441B")
  redPalette <- c("#FFF5F0","#FEE0D2","#FCBBA1","#FC9272","#FB6A4A","#EF3B2C","#CB181D","#A50F15","#67000D")
  getColor <- function (greenOrRed = "green", amount = 0) {
    if (amount == 0)
      return("#FFFFFF")
    palette <- greenPalette
    if (greenOrRed == "red")
      palette <- redPalette
    colorRampPalette(palette)(100)[10 + ceiling(90 * amount / total)]
  }
  
  # set the basic layout
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  classes = colnames(cm$table)
  rect(150, 430, 240, 370, col=getColor("green", res[1]))
  text(195, 435, classes[1], cex=1.2)
  rect(250, 430, 340, 370, col=getColor("red", res[3]))
  text(295, 435, classes[2], cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col=getColor("red", res[2]))
  rect(250, 305, 340, 365, col=getColor("green", res[4]))
  text(140, 400, classes[1], cex=1.2, srt=90)
  text(140, 335, classes[2], cex=1.2, srt=90)
  
  # add in the cm results
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}
draw_confusion_matrix(cm)
