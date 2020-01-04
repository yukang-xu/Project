# 1
data <- read.table("C:/Users/xuyuk/OneDrive - Georgia State University/Data import/vowel-train.txt",
                   header = TRUE, sep = ",", na.strings = 'null')
pca <- prcomp(data[,c(3:12)], center = TRUE,scale. = TRUE)
summary(pca)
library(factoextra)
fviz_eig(pca)


# 2
data1=data[,3:10]
R<-cor(data1)
e<-eigen(R)
zdat<-scale(data1)
pca.scores<- zdat %*% e$vectors
colnames(pca.scores)<-c('pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8')
head(pca.scores)
data2=cbind.data.frame(data[,2],pca.scores[,1:8])
colnames(data2)<-c('y','pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8')
library(MASS) 
lda <- lda(y ~ pca1+pca2+pca3+pca4+pca5+pca6, data = data2)
trainpred=predict(lda,data2)
mean(trainpred$class !=data2$y)
test <- read.table("C:/Users/xuyuk/OneDrive - Georgia State University/Data import/vowel-train.txt",
                   header = TRUE, sep = ",", na.strings = 'null')
test1=test[,3:10]
R1<-cor(test1)
e1<-eigen(R1)
zdat<-scale(test1)
pca.scores<- zdat %*% e1$vectors
colnames(pca.scores)<-c('pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8')
head(pca.scores)
test2=cbind.data.frame(test[,2],pca.scores[,1:8])
colnames(test2)<-c('y','pca1','pca2','pca3','pca4','pca5','pca6','pca7','pca8')
library(MASS) 
trainpred1=predict(lda,test2)
mean(trainpred1$class !=test2$y)


# 3
qda.model = qda (y ~ pca1+pca2+pca3+pca4+pca5+pca6+pca7+pca8, data = data2)
trainpred=predict(qda.model,data2)
mean(trainpred$class !=data2$y)
trainpred1=predict(qda.model,test2)
mean(trainpred1$class !=test2$y)


# 4
data=data[,2:12]
ldaori <- lda(y~., data = data)
qdaori= qda (y~., data = data)
test3=test[,2:12]
trainpred=predict(ldaori,data)
mean(trainpred$class !=data$y)
trainpred=predict(ldaori,test3)
mean(trainpred$class !=test3$y)
trainpred=predict(qdaori,data)
mean(trainpred$class !=data$y)
trainpred=predict(qdaori,test3)
mean(trainpred$class !=test3$y)


# 5 
trainpred2=predict(ldaori,data)
trainpred3=predict(ldaori,test3)
trainpred4=predict(qdaori,data)
trainpred5=predict(qdaori,test3)
errorline1=data[which(trainpred2$class !=data$y),] 
errorline2=test3[which(trainpred3$class !=test3$y),] 
errorline3=data[which(trainpred4$class !=data$y),] 
errorline4=test3[which(trainpred5$class !=test3$y),] 
a=rbind(errorline1,errorline2,errorline3,errorline4)
table(a[,1])
data3=subset(data, y!=2 & y!=6 & y!=9)
data4=subset(test3, y!=2 & y!=6 & y!=9)
ldaori <- lda(y~., data = data3)
qdaori= qda (y~., data = data3)
trainpred=predict(ldaori,data3)
mean(trainpred$class !=data3$y)
trainpred=predict(ldaori,data4)
mean(trainpred$class !=data4$y)
trainpred=predict(qdaori,data3)
mean(trainpred$class !=data3$y)
trainpred=predict(qdaori,data4)
mean(trainpred$class !=data4$y)


# 6
library(dendextend)
library(cluster)    
library(factoextra)
data5=subset(data, y!=2 & y!=4 & y!=5 & y!=7 & y!=8 & y!=9 & y!=11)
data6=subset(test3, y!=2 & y!=4 & y!=5 & y!=7 & y!=8 & y!=9 & y!=11)
# hierarchical clustering analysis
d <- dist(data5, method = "euclidean")
hc1 <- hclust(d, method = "complete" )
plot(hc1, cex = 0.6, hang = -1)
hc5 <- hclust(d, method = "ward.D2" )
sub_grp <- cutree(hc5, k = 4)
fviz_cluster(list(data = data5, cluster = sub_grp))
# K-mean method
k2 <- kmeans(scale(data5), centers = 4, nstart = 25)
k2$cluster <- as.factor(k2$cluster)
ggplot(data5, aes(x.1, x.2, color = k2$cluster)) + geom_point()
# Model-based clustering
fit <- Mclust(data5)
summary(fit)
# assessment of clustering
data5$y[data5$y == 3] <- 2
data5$y[data5$y == 6] <- 3
data5$y[data5$y == 10] <- 4
table(data5[,1],k2$cluster)
table(data5[,1],sub_grp)
