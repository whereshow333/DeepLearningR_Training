# # # 安裝套件步驟 # # # 

# 1.安裝R接Python的套件
install.packages('reticulate')
require(reticulate)


# 2.安裝devtools
install.packages('devtools')


# 3.從github安裝keras
# 安裝時也會把tensorflow一起裝好
# 若沒有裝好可以先裝好tensorflow,參考網址：https://tensorflow.rstudio.com
devtools::install_github("rstudio/keras") 


# 4.載入keras,安裝Tensorflow engine
library(keras)
install_tensorflow()





# # # 小小的手寫辨識範例 # # # 
library(keras)

# 使用範例資料mnist
data<-dataset_mnist()

#Training Data
# 60,000個28x28灰度圖像,圖像是手寫辨識0~9
train_x<-data$train$x
train_y<-data$train$y

#Test Set
# 10,000個28x28灰度圖像,圖像是手寫辨識0~9
test_x<-data$test$x
test_y<-data$test$y

# 把每一張28x28的圖片拉成1x784的向量
# 再把60000個向量放在一起變成一個60000x784的矩陣
train_x <- array(as.numeric(train_x), dim = c(dim(train_x)[[1]], 784))

# 把每一張28x28的圖片拉成1x784的向量
# 再把10000個向量放在一起變成一個10000x784的矩陣
test_x <- array(as.numeric(test_x), dim = c(dim(test_x)[[1]], 784))

# 圖片像素標準化
train_x <- train_x / 255
test_x <- test_x / 255

#60000 train examples
cat(dim(train_x)[[1]], 'train samples\n')

#10000 test examples
cat(dim(test_x)[[1]], 'test samples\n')

#將手寫辨識的0~9變成類別型態
train_y<-to_categorical(train_y,10)
test_y<-to_categorical(test_y,10)





model <- keras_model_sequential()

model %>%

# 第一層
# dropout layer 避免過度配適
layer_dense(units = 512, activation = 'relu', input_shape = c(784)) %>%
layer_dropout(rate = 0.2) %>%

# 第二層
layer_dense(units = 512, activation = 'relu') %>%
layer_dropout(rate = 0.2) %>%

# 第三層,輸出層
layer_dense(units = 10, activation = 'softmax')


summary(model)


# 編譯模型,最佳化方法使用sgd
# metrics是評估模型的指標
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_sgd(lr = 0.02),
  metrics = c('accuracy')
)



# 訓練模型
#epochs = 迭代的次數
#batchsize = 每一次梯度更新的樣本數
history <- model %>% fit(train_x, train_y,epochs = 20, batch_size = 128,validation_split = 0.2)


summary(history)

history$params

# 每一次迭代的模型評估指標
history$metrics

# 畫出每次迭代與準確率的圖
# 畫出每次迭代與損失函數的圖
plot(history)


plot(x = history$metrics$acc,y = history$metrics$loss,pch=19,col='red',type='b',ylab="Error on trining Data",xlab="Accuracy on Training Data")
title("Plot of accuracy vs Loss")
legend("topright",c("Epochs"),col="red",pch=19)



# 計算模型的損失函數與模型評估指標
loss_and_metrics <- model %>% evaluate(test_x, test_y, batch_size = 128)

# # > loss_and_metrics
# # [[1]]
# # [1] 0.1210407

# # [[2]]
# # [1] 0.9627



# 拿測試集來預測
classes <- model %>% predict(test_x, batch_size = 128)