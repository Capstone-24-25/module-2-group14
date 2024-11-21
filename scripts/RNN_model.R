# can comment entire section out if no changes to preprocessing.R
source('scripts/preprocessing.R')

# load raw data
load('data/claims-raw.RData')

# preprocess (will take a minute or two)
claims_clean <- claims_raw %>%
  parse_data()

# export
save(claims_clean, file = 'data/claims-clean-example.RData')

# Load required libraries
library(keras)
library(tensorflow)
library(tidyverse)
library(tidymodels)

# Load preprocessed data
load('data/claims-clean-example.RData')

# Partition data into training and testing sets
set.seed(110122)
partitions <- claims_clean %>%
  initial_split(prop = 0.8)

train_text <- training(partitions) %>%
  pull(text_clean)
train_labels_binary <- training(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1
train_labels_multi <- training(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

test_text <- testing(partitions) %>%
  pull(text_clean)
test_labels_binary <- testing(partitions) %>%
  pull(bclass) %>%
  as.numeric() - 1
test_labels_multi <- testing(partitions) %>%
  pull(mclass) %>%
  as.numeric() - 1

# Define vocabulary size and sequence length
vocab_size <- 10000
sequence_length <- 100

# Create and adapt the text vectorization layer
preprocess_layer <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_sequence_length = sequence_length
)
preprocess_layer %>% adapt(train_text)

# Preprocess data
train_sequences <- preprocess_layer(train_text)
test_sequences <- preprocess_layer(test_text)

# Define model for binary classification
binary_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, input_length = sequence_length) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# Compile binary model
binary_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# Train binary model
binary_model %>% fit(
  x = train_sequences,
  y = train_labels_binary,
  validation_split = 0.2,
  epochs = 10,
  batch_size = 32
)

# Save binary model
save_model_tf(binary_model, "results/binary-model")

# Define model for multi-class classification
multi_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, input_length = sequence_length) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = length(unique(train_labels_multi)), activation = 'softmax')

# Compile multi-class model
multi_model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'sparse_categorical_accuracy'
)

# Train multi-class model
multi_model %>% fit(
  x = train_sequences,
  y = train_labels_multi,
  validation_split = 0.2,
  epochs = 10,
  batch_size = 32
)

# Save multi-class model
save_model_tf(multi_model, "results/multi-model")

# Generate predictions
binary_preds <- binary_model %>% predict(test_sequences) %>% round()

# Ensure multi_preds is properly indexed
multi_preds <- as.numeric(multi_preds)  # Convert TensorFlow predictions to numeric

# Format predictions into a data frame
pred_df <- testing(partitions) %>%
  select(.id) %>%
  mutate(
    bclass.pred = ifelse(binary_preds == 1, "Positive", "Negative"),
    mclass.pred = levels(factor(claims_clean$mclass))[multi_preds + 1]
  )

# Export predictions
write.csv(pred_df, "results/example-preds.csv", row.names = FALSE)

# Display results
print(head(pred_df))