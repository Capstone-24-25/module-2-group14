
# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

## MODEL TRAINING (RNN)
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

# define vocabulary size and sequence length
vocab_size <- 10000  # Maximum number of words in the vocabulary
sequence_length <- 100  # Length of sequences (padding/truncation)

# create a text vectorization layer
preprocess_layer <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_sequence_length = sequence_length
)

# adapt the layer on the training data
preprocess_layer %>% adapt(train_text)

# vectorize the text data
train_sequences <- preprocess_layer(train_text)

# define the RNN model architecture
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, input_length = sequence_length) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%  # LSTM layer
  layer_dropout(0.5) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')  # Output layer for binary classification

# summarize the model architecture
summary(model)

# compile the model
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# train the model
history <- model %>%
  fit(
    x = train_sequences,
    y = train_labels,
    validation_split = 0.2,
    epochs = 10,  # Number of epochs
    batch_size = 32  # Batch size
  )

## EVALUATE ON TEST SET
test_text <- testing(partitions) %>%
  pull(text_clean)
test_labels <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1

# preprocess test data
test_sequences <- preprocess_layer(test_text)

# evaluate model
metrics <- model %>% evaluate(test_sequences, test_labels)

# print test set accuracy
print(paste("Test set accuracy:", metrics["binary_accuracy"]))

# save the entire model
save_model_tf(model, "results/example-rnn-model")