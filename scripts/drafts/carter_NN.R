## PREPROCESSING
#################

# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

## MODEL TRAINING (NN)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

# load cleaned data
load('data/claims-clean-example.RData')

# partition
set.seed(27)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# If having library conflicts
#install.packages("keras", type = "source")
#library(keras)
#install_keras()

# create a preprocessing layer
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)

preprocess_layer %>% adapt(train_text)

# define NN architecture
model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dense(units = 64, activation = "sigmoid") %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.15) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

# configure for training
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# train
history <- model %>%
  fit(train_text, 
      train_labels,
      validation_split = 0.3,
      epochs = 5)

## CHECK TEST SET ACCURACY
##########################
# Extract test data
test_text <- testing(partitions) %>%
  pull(text_clean)
test_labels <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# Get predictions
predictions <- model %>%
  predict(test_text) %>%
  as.vector()

# Convert probabilities to binary predictions
predicted_labels <- ifelse(predictions > 0.5, 1, 0)

# Calculate accuracy
test_accuracy <- mean(predicted_labels == test_labels)
cat('Test Set Accuracy:', test_accuracy, '\n')

# Test Set Accuracy: 0.8130841

## SAVING MODEL
###############
# save the entire model as a SavedModel
save_model_tf(model, "results/nn-model")
