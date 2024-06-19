
# Preprocessing
library(kohonen)
library(dummies)
library(ggplot2)
library(sp)
library(maptools)
library(reshape2)
library(rgeos)
library(sf)
library(terra)
library(MASS)
library(Hmisc)
library(RSNNS)
Sys.setenv(LANG = "en")

# Colour palette definition
pretty_palette <- c("#1f77b4", '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
'#8c564b', '#e377c2')
### DATA PREPARATION
data <- read.csv("./creditworthiness.csv")
describe(data)
classifiedData = subset(data, data[,46] > 0)
unknownData = subset(data, data[,46] == 0)
corTable = abs(cor(classifiedData, y=classifiedData$credit.rating))
corTable = corTable[order(corTable, decreasing = TRUE),,drop = FALSE]
head(corTable,6)


# ------------------- SOM TRAINING ---------------------------
#choose the variables with which to train the SOM
#the following selects column 1,2,3,4,6
interestedFeatures <- data[, c(1,2,3,4,6)]
#data_train <- data[, c(1:45)]
data_train <- classifiedData[, c(1:45)]
# now train the SOM using the Kohonen method
data_train_matrix <- as.matrix(scale(data_train))
names(data_train_matrix) <- names(data_train)
require(kohonen)
x_dim=20
y_dim=20
small_areas <-FALSE
if (small_areas){
# larger grid for the small areas example (more samples)
som_grid <- somgrid(xdim = x_dim, ydim=y_dim, topo="hexagonal")
} else {
som_grid <- somgrid(xdim = x_dim/2, ydim=y_dim/2, topo="hexagonal")
}
# Train the SOM model!
if (packageVersion("kohonen") < 3){
system.time(som_model <- som(data_train_matrix,
grid=som_grid,
rlen=1000,
alpha=c(0.8,0.01),
n.hood = "circular",
keep.data = TRUE ))
}else{
system.time(som_model <- som(data_train_matrix,
grid=som_grid,
rlen=1000,
alpha=c(0.8,0.01),
mode="online",
normalizeDataLayers=false,
keep.data = TRUE ))
}
summary(som_model)
rm(som_grid, data_train_matrix)


# -------------------- SOM VISUALISATION -----------------
source('./coolBlueHotRed.R')
# Plot the heatmap for a variable at scaled / normalised values
var <- 1 # Functionary
var_unscaled <- aggregate(as.numeric(data_train[,var]),
by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,2]
plot(som_model, type = "property", property=var_unscaled,
main=names(data_train)[var], palette.name=coolBlueHotRed)
rm(var_unscaled, var)
var <- 2 # FI3O.credit.score
var_unscaled <- aggregate(as.numeric(data_train[,var]),
by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,2]
plot(som_model, type = "property", property=var_unscaled,
main=names(data_train)[var], palette.name=coolBlueHotRed)
rm(var_unscaled, var)
var <- 3 # Rebalance.payback
var_unscaled <- aggregate(as.numeric(data_train[,var]),
by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,2]
plot(som_model, type = "property", property=var_unscaled,
main=names(data_train)[var], palette.name=coolBlueHotRed)
rm(var_unscaled, var)
var <- 4 # credit.refused.in.past.
var_unscaled <- aggregate(as.numeric(data_train[,var]),
by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,2]
plot(som_model, type = "property", property=var_unscaled,
main=names(data_train)[var], palette.name=coolBlueHotRed)
rm(var_unscaled, var)
var <- 6 #Gender
var_unscaled <- aggregate(as.numeric(data_train[,var]),
by=list(som_model$unit.classif), FUN=mean, simplify=TRUE)[,2]
plot(som_model, type = "property", property=var_unscaled,
main=names(data_train)[var], palette.name=coolBlueHotRed)
rm(var_unscaled, var)
source('./plotHeatMap.R')
plotHeatMap(som_model, classifiedData, variable=0)
genderT = with(classifiedData, table(credit.rating, gender))
barplot(genderT, beside = TRUE,
legend = c("Credit Rating A", "Credit Rating B", "Credit Rating
C"),
col = c("darkgreen","yellow", "red"),
main = "Gender vs Credit Rating",
sub="0 = Male, 1 = Female")
selfEmployed = with(classifiedData, table(credit.rating, self.employed.))
barplot(selfEmployed, beside = TRUE,
legend = c("Credit Rating A", "Credit Rating B", "Credit Rating
C"),
col = c("darkgreen","yellow", "red"),
main = "Self Employed vs Credit Rating",
sub="0 = No, 1 = Yes")
functional = with(classifiedData, table(credit.rating, functionary))
barplot(functional, beside = TRUE,
legend = c("Credit Rating A", "Credit Rating B", "Credit Rating
C"),
col = c("darkgreen","yellow", "red"),
main = "Functionary vs Credit Rating",
sub="0 = No, 1 = Yes")
genderT = with(classifiedData, table(credit.rating, gender))
genderT
barplot(genderT, beside = TRUE,
legend = c("Credit Rating A", "Credit Rating B", "Credit Rating
C"),
col = c("darkgreen","yellow", "red"),
main = "Gender vs Credit Rating",
sub="0 = Male, 1 = Female")
selfEmployed = with(classifiedData, table(credit.rating, self.employed.))
barplot(selfEmployed, beside = TRUE,
legend = c("Credit Rating A", "Credit Rating B", "Credit Rating
C"),
col = c("darkgreen","yellow", "red"),
main = "Self Employed vs Credit Rating",
sub="0 = No, 1 = Yes")
# generate a contingency table and plot a barplot
FI30T = with(classifiedData, table(credit.rating, FI3O.credit.score))
FI30T
barplot(genderT, beside = TRUE,
legend = c("Credit Rating A", "Credit Rating B", "Credit Rating
C"),
col = c("darkgreen","yellow", "red"),
main = "FI30 vs Credit Rating",
sub="0 = Not OK, 1 = Ok")
# show the WCSS metric for kmeans for different clustering sizes.
# Can be used as a "rough" indicator of the ideal number of clusters
mydata <- matrix(unlist(som_model$codes), ncol = length(data_train),
byrow = FALSE)
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(mydata,
centers=i)$withinss)
par(mar=c(5.1,4.1,4.1,2.1))
plot(1:15, wss, type="b", xlab="Number of Clusters",
ylab="Within groups sum of squares", main="Within cluster sum of
squares (WCSS)")
# Form clusters on grid
## use hierarchical clustering to cluster the codebook vectors
som_cluster <- cutree(hclust(dist(mydata)), 3)
# Show the map with different colours for every cluster
plot(som_model, type="mapping", bgcol = pretty_palette[som_cluster], main
= "Clusters")
add.cluster.boundaries(som_model, som_cluster)
#show the same plot with the codes instead of just colours
plot(som_model, type="codes", bgcol = pretty_palette[som_cluster], main =
"Clusters")
add.cluster.boundaries(som_model, som_cluster)


# ------------------ Clustering SOM results -------------------
# show the WCSS metric for kmeans for different clustering sizes.
# Can be used as a "rough" indicator of the ideal number of clusters
mydata <- matrix(unlist(som_model$codes), ncol = length(data_train),
byrow = FALSE)
wss <- (nrow(mydata)-1)*sum(apply(mydata,2,var))
for (i in 2:15) wss[i] <- sum(kmeans(mydata,
centers=i)$withinss)
par(mar=c(5.1,4.1,4.1,2.1))
plot(1:15, wss, type="b", xlab="Number of Clusters",
ylab="Within groups sum of squares", main="Within cluster sum of
squares (WCSS)")
# Form clusters on grid
## use hierarchical clustering to cluster the codebook vectors
som_cluster <- cutree(hclust(dist(mydata)), 3)
# Show the map with different colours for every cluster
plot(som_model, type="mapping", bgcol = pretty_palette[som_cluster], main
= "Clusters")
add.cluster.boundaries(som_model, som_cluster)
#show the same plot with the codes instead of just colours
plot(som_model, type="codes", bgcol = pretty_palette[som_cluster], main =
"Clusters")
add.cluster.boundaries(som_model, som_cluster)
# To train the MLP model to classified based on the following
# interested columns.
interestedColumns = c(1, 2, 3, 4, 6, 9)
# Seperate value from targets
trainValues = classifiedData[,interestedColumns]
unknownValues = unknownData[,interestedColumns]
# Use decodeClassLabels() to decode class labels from a
# numerical or levels vector to a binary matrix.
trainTargets = decodeClassLabels(classifiedData[,46])
# Split the data into training and testing data set
trainSet = splitForTrainingAndTest(trainValues, trainTargets, ratio =
0.2)
# Normalized the training data set
trainSet = normTrainingAndTestSet(trainSet)
# Train the MLP model
model = mlp(trainSet$inputsTrain,
trainSet$targetsTrain,
size=c(20),
learnFuncParams=c(0.001),
maxit=100,
inputsTest = trainSet$inputsTest,
targetsTest = trainSet$targetsTest)
#
# Predict the test
# The predict() function in R is used to predict the values
# based on the input data.
predictTestSet = predict(model, trainSet$inputsTest)
# Predict the unknown set
predictUnknownSet = predict(model, unknownValues)
# Compute the confusion matrix
confusionMatrix(trainSet$targetsTrain, fitted.values(model))
confusionMatrix(trainSet$targetsTest, predictTestSet)
# interpreting the unknown data set (prediction)
head(trainTargets)
head(classifiedData[,46])
head(predictUnknownSet)
# Plot
par(mar=c(5.1,4.1,4.1,2.1))
par(mfrow=c(2,2))
plotIterativeError(model)
plotRegressionError(predictTestSet[,2], trainSet$targetsTest[,2])
plotROC(fitted.values(model)[,2], trainSet$targetsTrain[,2])
plotROC(predictTestSet[,2],trainSet$targetsTest[,2])
summary(model)
model
weightMatrix(model)
extractNetInfo(model)