# Introduction to Artificial Neural Networks (ANN)

## Applications of ANNs

ANNs are applied in a wide range of real-world scenarios, some of which include:

# install.packages("dplyr")
# install.packages("neuralnet")
library(dplyr)
library(neuralnet)

## 2. Obtain Data

exam<-c(20,10,30,20,80,30)
proyect<-c(90,20,40,50,50,80)
status<-c(1,0,0,0,0,1)

df<-data.frame(exam,proyect,status)


## 3. Generate Neural Network
nn1 <- neuralnet(status ~., data=df)
plot(nn1, rep="best")

## <span style="color: blue;">4. Predict Results</span>
Now that our neural network model is trained, we proceed to evaluate its predictive performance using a new set of exam and project scores. This step is essential for assessing the model's ability to generalize and make accurate predictions on data it hasn't seen before. By inputting these new scores into the model, we aim to predict their corresponding statuses, providing insights into how well our model can apply learned patterns to real-world or hypothetical scenarios. This evaluation phase is crucial for understanding the practical applicability of our neural network in predicting outcomes based on exam and project performances.


exam_test<-c(30,40,85)
proyect_test<-c(85,50,40)
test1 <- data_frame(exam_test,proyect_test)
prediction <- compute(nn1,test1)

prediction$net.result


## 1. Install Packages and Call Libraries

# install.packages("readr")
# install.packages("caret")
library(readr)
library(caret)


## 2. Obtain Data
breast_cancer <- read_csv("cancer_de_mama.csv")
breast_cancer$diagnosis <- ifelse(breast_cancer$diagnosis == "M", 1, 0)

# Create indices for a 75% train and 25% test split
set.seed(123) # Setting seed for reproducibility
trainIndex <- createDataPartition(breast_cancer$diagnosis, p = 0.75, 
                                  list = FALSE, 
                                  times = 1)

# Split the data into training and testing sets
trainSet <- breast_cancer[trainIndex, ]
testSet <- breast_cancer[-trainIndex, ]


## 3. Clean Data
names(trainSet) <- make.names(names(trainSet), unique = TRUE)

names(testSet) <- make.names(names(trainSet), unique = TRUE)
testSetBlind <- subset(testSet, select = -diagnosis)


## 4. Generate Neural Network
nn2 <- neuralnet(diagnosis ~ ., data=trainSet, hidden=c(5), linear.output=FALSE)
plot(nn2, rep="best")


## 5. Predict Results
prediction <- compute(nn2,testSetBlind)

predicted_labels <- ifelse(prediction$net.result > 0.5, 1, 0)
true_labels <- testSet$diagnosis
conf_matrix <- table(Predicted = predicted_labels, Actual = true_labels)

# Accuracy Metrics
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
recall <- conf_matrix[2,2] / sum(conf_matrix[2,])
specificity <- conf_matrix[1,1] / sum(conf_matrix[1,])

cat("Accuracy:", accuracy, "\n")
cat("Recall:", recall, "\n")
cat("Specificity:", specificity, "\n")
