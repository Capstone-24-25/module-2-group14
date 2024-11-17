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
set.seed(110122)
preprocess_layer <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = NULL,
  output_mode = 'tf_idf'
)

preprocess_layer %>% adapt(train_text)

# define NN architecture
set.seed(110122)
model <- keras_model_sequential() %>%
  preprocess_layer() %>%
  layer_dropout(0.22) %>%
  layer_dense(units = 25) %>%
  layer_dropout(0.16) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model)

# configure for training
set.seed(110122)
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

# Test Set Accuracy: 0.8014019 

## SAVING MODEL
###############
# save the entire model as a SavedModel
save_model_tf(model, "results/nn-model")


## MODEL TRAINING (LPCA)
######################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)
library(logisticPCA)
library(dplyr)
library(textrecipes)

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

# Load cleaned data
load('data/claims-clean-example.RData')

# Partition data
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_data <- training(partitions)
test_data <- testing(partitions)

# Define preprocessing recipe (words)
recipe <- recipe(bclass ~ text_clean, data = train_data) %>%
  step_tokenize(text_clean) %>%
  step_stopwords(text_clean) %>%
  step_tokenfilter(text_clean, max_tokens = 5000) %>%
  step_tfidf(text_clean)

# Prepare the recipe
prepared_recipe <- prep(recipe, training = train_data)

# Extract the TF-IDF matrix for logistic PCA
train_matrix <- bake(prepared_recipe, new_data = train_data) %>%
  select(-bclass) %>%
  as.matrix()

test_matrix <- bake(prepared_recipe, new_data = test_data) %>%
  select(-bclass) %>%
  as.matrix()

train_labels <- train_data$bclass
test_labels <- test_data$bclass

# Perform PCA on the TF-IDF matrix
pca_result <- prcomp(train_matrix, scale. = TRUE)

# Choose the number of components to keep (e.g., 10 components)
num_components <- 10
train_pca <- pca_result$x[, 1:num_components]

# Apply PCA transformation on the test data
test_pca <- predict(pca_result, newdata = test_matrix)[, 1:num_components]

# Convert labels to factors for logistic regression
train_labels <- as.factor(train_labels)
test_labels <- as.factor(test_labels)

# Create a tibble for model fitting
train_data_pca <- as_tibble(train_pca) %>%
  mutate(bclass = train_labels)

test_data_pca <- as_tibble(test_pca) %>%
  mutate(bclass = test_labels)

# Define logistic regression model
log_reg_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Fit logistic regression model
log_reg_fit <- log_reg_model %>%
  fit(bclass ~ ., data = train_data_pca)

# CHECK TEST SET ACCURACY
##########################
# Predict on test set
test_predictions <- log_reg_fit %>%
  predict(new_data = test_data_pca, type = "prob") %>%
  bind_cols(test_data_pca)

# Convert probabilities to binary predictions
test_predictions <- test_predictions %>%
  mutate(pred_class = ifelse(`.pred_Relevant claim content` > 0.5, 1, 0),
         pred_class = as.factor(pred_class),
         bclass_numeric = ifelse(bclass == "Relevant claim content", 1, 0)
  )

# Calculate accuracy
accuracy <- mean(test_predictions$pred_class == test_predictions$bclass_numeric)
cat('Test Set Accuracy:', accuracy, '\n')

# Define preprocessing recipe (bigrams)
recipe <- recipe(bclass ~ text_clean, data = train_data) %>%
  step_tokenize(text_clean, token = "ngrams") %>%
  step_ngram(text_clean, min_num_tokens = 2, num_tokens = 2) %>%
  step_stopwords(text_clean) %>%
  step_tokenfilter(text_clean, max_tokens = 5000) %>%
  step_tfidf(text_clean)

# Prepare the recipe
prepared_recipe <- prep(recipe, training = train_data)

# Extract the TF-IDF matrix for logistic PCA
train_matrix <- bake(prepared_recipe, new_data = train_data) %>%
  select(-bclass) %>%
  as.matrix()

test_matrix <- bake(prepared_recipe, new_data = test_data) %>%
  select(-bclass) %>%
  as.matrix()

train_labels <- train_data$bclass
test_labels <- test_data$bclass

# Perform PCA on the TF-IDF matrix
pca_result <- prcomp(train_matrix, scale. = TRUE)

# Choose the number of components to keep (e.g., 10 components)
num_components <- 10
train_pca <- pca_result$x[, 1:num_components]

# Apply PCA transformation on the test data
test_pca <- predict(pca_result, newdata = test_matrix)[, 1:num_components]

# Convert labels to factors for logistic regression
train_labels <- as.factor(train_labels)
test_labels <- as.factor(test_labels)

# Create a tibble for model fitting
train_data_pca <- as_tibble(train_pca) %>%
  mutate(bclass = train_labels)

test_data_pca <- as_tibble(test_pca) %>%
  mutate(bclass = test_labels)

# Define logistic regression model
log_reg_model <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

# Fit logistic regression model
log_reg_fit <- log_reg_model %>%
  fit(bclass ~ ., data = train_data_pca)

# CHECK TEST SET ACCURACY
##########################
# Predict on test set
test_predictions <- log_reg_fit %>%
  predict(new_data = test_data_pca, type = "prob") %>%
  bind_cols(test_data_pca)

# Convert probabilities to binary predictions
test_predictions <- test_predictions %>%
  mutate(pred_class = ifelse(`.pred_Relevant claim content` > 0.5, 1, 0),
         pred_class = as.factor(pred_class),
         bclass_numeric = ifelse(bclass == "Relevant claim content", 1, 0)
         )

# Calculate accuracy
accuracy <- mean(test_predictions$pred_class == test_predictions$bclass_numeric)
cat('Test Set Accuracy:', accuracy, '\n')

# Test Set Accuracy: 0.7079439 without header
# Test Set Accuracy: 0.7149533 with header
# Test Set Accuracy: 0.5373832 with bigram

