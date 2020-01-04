# Import data
data=read.csv("C:/Users/xuyuk/OneDrive - Georgia State University/Data import/framingham.csv")

# Fit model
fit<-glm(TenYearCHD~.,data=data,family="binomial")
summary(fit)

# Feature selection
data=data[,-(3:4)]
data=data[,-(4:7)]
data=data[,-(5:8)]

# Remove missing value
data=data[complete.cases(data), ]
fit<-glm(TenYearCHD~.,data=data,family="binomial")
summary(fit)

# Model selection
full<-glm(TenYearCHD~male*age*cigsPerDay*totChol*glucose,data=data,family="binomial")
null<-glm(TenYearCHD~1,data=data,family=binomial)
step(null,scope=list(lower=null,upper=full),direction="both")
fit1=glm(formula=TenYearCHD~age+cigsPerDay+glucose + male + totChol + age:totChol + glucose:totChol, family = binomial, data = data)
summary(fit1)

# Remove insignificant interaction term
fit2=glm(TenYearCHD~male+age+cigsPerDay+totChol+glucose,data=data,family="binomial")
summary(fit2)

# Convert the coefficients to odds-ratios
exp(coef(fit2))

# Create a confidence interval of odds-ratios
exp(cbind(OR=coef(fit2),confint(fit2)))

# Anova Test to Determine Goodness of Fit
anova(fit2,test="Chisq")

# Cook's distance
cooks.distance<-cooks.distance(fit2)
which(cooks.distance>1)

# Wald Test to determine if predictors are significant
library(survey)
regTermTest(fit2,"male")
regTermTest(fit2,"age")
regTermTest(fit2,"CigsPerDay")
regTermTest(fit2,"totChol")
regTermTest(fit2,"glucose")

# Hoslem-Lemeshow Goodness of Fit Test
library(ResourceSelection)
hoslem.test(fit2$y,fitted(fit2),g=10)

# Looking at VIF for Collinearity
library(car)
vif(fit2)

# Determining the Pseudo-Rsq
library(pscl)
pR2(fit2)

# Plotting the effects of age, sex, and class to predict ten year CHD
library(effects)
plot(allEffects(fit2))

# Cross Validation to obtain accuracy of model
library(caret)
library(plyr)
ctrl<-trainControl(method="repeatedcv",number=10,savePredictions=TRUE)
mod_fit<-train(TenYearCHD~male+age+cigsPerDay+totChol+glucose,data=data,method="glm",family="binomial",trControl=ctrl,tuneLength=5)
Train<-createDataPartition(data$TenYearCHD,p=0.8,list=FALSE)
training<-data[Train,]
testing<-data[-Train,]
y_testing=testing[,6]
x_testing=testing[,1:5]
prob <- predict(mod_fit, newdata=testing, type="raw")
results <- ifelse(prob > 0.5,1,0)
results=as.factor(results)
y_testing=as.factor(y_testing)
confusionMatrix(data=results,y_testing)

# Determining Variables of Importance
varImp(mod_fit)

# Graphing and finding the area underneath the ROC Curve:
library(ROCR)
p<-predict(fit2,newdata=subset(testing,select=c(1,2,3,4,5)),type="response")
pr<-prediction(p,testing$TenYearCHD)
prf<-performance(pr,measure="tpr",x.measure="fpr")
plot(prf)
auc<-performance(pr,measure="auc")
auc<-auc@y.values[[1]]
auc


