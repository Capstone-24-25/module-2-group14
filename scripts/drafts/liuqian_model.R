## PREPROCESSING
################

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
set.seed(110122)
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
  layer_dropout(0.2) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

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

## SAVING MODEL
###############
# save the entire model as a SavedModel
save_model_tf(model, "results/example-model")

claims_clean %>%
  add_predictions(fit, type = 'response') %>%
  mutate(est = as.factor(pred > 0.5), tr_c = as.factor(class)) %>%
  class_metrics(estimate = est,
                truth = tr_c, pred,
                event_level = 'second')


## MODEL TRAINING (LPCA)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)
library(logisticPCA)
library(dplyr)

# path to activity files on repo
url <- 'https://raw.githubusercontent.com/pstat197/pstat197a/main/materials/activities/data/'

# load a few functions for the activity
source(paste(url, 'projection-functions.R', sep = ''))

# load cleaned data
load('data/claims-clean-example.RData')

# partition
set.seed(110123)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

# separate DTM from labels
test_dtm <- testing(partitions) %>%
  select(-.id, -bclass, -mclass)
test_labels <- testing(partitions) %>%
  select(.id, bclass, mclass)

# same, training set
train_dtm <- training(partitions) %>%
  select(-.id, -bclass, -mclass)
train_labels <- training(partitions) %>%
  select(.id, bclass, mclass)

# find projections based on training data
proj_out <- projection_fn(.dtm = train_dtm, .prop = 0.7)
train_dtm_projected <- proj_out$data

# how many components were used?
proj_out$n_pc
# CHECK TEST SET ACCURACY
##########################
# Get predictions
test_results <- log_reg_fit %>%
  predict(test_data, type = "prob") %>%
  bind_cols(test_data)

# Convert probabilities to binary predictions
test_results <- test_results %>%
  mutate(pred_class = ifelse(.pred_1 > 0.5, 1, 0))

# Calculate Accuracy
accuracy <- mean(test_results$pred_class == as.numeric(test_data$bclass) - 1)
cat('Test Set Accuracy:', accuracy, '\n')


