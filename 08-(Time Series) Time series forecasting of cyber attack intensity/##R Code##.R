##########################
# Load the package 
install.packages("RJSONIO")
install.packages("highfrequency")
library("RJSONIO")
library("highfrequency")
library(xts)
library('stats')
library(matrixStats)
library(forecast)
library(tseries)

# Import data
json_file = 'C:/Users/xuyuk/Documents/failuretime.json'
json_file = RJSONIO::fromJSON(json_file)
data = as.data.frame(json_file)
data1='null'
data1$time=as.POSIXlt(data$time)
data1$count=seq(1,1,length.out=93142)
data1=as.data.frame(data1)
data1=data1[,-1]
str(data1)

# Visualize our data
data1$cut=cut(data1$time, breaks="15 mins")   
tsagg15min = aggregate(count~cut,FUN='sum',data = data1);
head(tsagg15min)
plot.ts(tsagg15min$count)

# Stationary test
count=tsagg15min$count
ndiffs(count)
diff=diff(count)
plot(diff,xlab="time period",ylab="timestamps")
adf.test(diff)

# Model selection
acf(diff,main='ACF')
pacf(diff,main='PACF')
auto.arima(count)

# Fitting the model
fit = arima(count, order = c(0, 1, 1))

# Model evaluation
qqnorm(fit$residuals)
qqline(fit$residuals)
Box.test(fit$residuals,type="Ljung-Box")
accuracy(fit)

# Prediction
forecast=forecast(fit,4)
plot(forecast(fit,4),xlab="time period",ylab="timestamps")
plot(forecast(fit,4),xlim=c(9300,9390),xlab="time period",ylab="timestamps")
###################
