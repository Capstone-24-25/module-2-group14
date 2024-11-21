
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

# Create and adapt the text vectorization layer
preprocess_layer <- layer_text_vectorization(
  max_tokens = 10000,
  output_sequence_length = 100
)
preprocess_layer %>% adapt(train_text)

# Preprocess data
train_sequences <- preprocess_layer(train_text)
test_sequences <- preprocess_layer(test_text)

# Define model for binary classification
binary_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 128, input_length = 100) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32) %>%
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
  epochs = 11,
  batch_size = 32
)

# Save binary model
save_model_tf(binary_model, "results/binary-model")

# Define model for multi-class classification
multi_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = 10000, output_dim = 128, input_length = 100) %>%
  layer_lstm(units = 64, return_sequences = FALSE) %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 32) %>%
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
  epochs = 11,
  batch_size = 32
)

# Save multi-class model
save_model_tf(multi_model, "results/multi-model")

# Generate predictions
binary_preds <- binary_model %>% predict(test_sequences) %>% round()

multi_preds <- multi_model %>% predict(test_sequences) %>% k_argmax()

# Format predictions into a data frame
class_labels <- levels(factor(claims_clean$mclass))

pred_df <- testing(partitions) %>%
  select(.id) %>%
  mutate(
    bclass.pred = ifelse(binary_preds == 1, "Positive", "Negative"),
    mclass.pred = class_labels[as.numeric(multi_preds) + 1]  # Correct indexing
  )

# Export predictions
write.csv(pred_df, "results/preds_df.csv", row.names = FALSE)

# Display results
print(head(pred_df))

### Predictions
### Using Claims-test

# Load trained RNN models
binary_model <- load_model_tf("results/binary-model")
multi_model <- load_model_tf("results/multi-model")

load('data/claims-test.RData')

# Preprocess claims-test data
clean_df <- claims_test %>%
  parse_data() %>%  # Apply same preprocessing pipeline used for training
  select(.id, text_clean)


# Prepare test input
x_test <- clean_df %>%
  pull(text_clean)

# Convert text to numerical sequences using the trained preprocessing layer
test_sequences <- preprocess_layer(as.array(x_test))

# Generate binary predictions
binary_preds <- binary_model %>% predict(test_sequences) %>% round() 

# Generate multi-class predictions
multi_preds <- multi_model %>% predict(test_sequences) %>% k_argmax()

# Map predictions to class labels
class_labels_binary <- c("Negative", "Positive")  
class_labels_multi <- levels(factor(claims_clean$mclass))  

binary_pred_classes <- factor(binary_preds, labels = class_labels_binary)
multi_pred_classes <- factor(as.numeric(multi_preds) + 1, labels = class_labels_multi)

# Format predictions into a data frame
pred_df <- clean_df %>%
  select(.id) %>%
  mutate(
    bclass.pred = binary_pred_classes,
    mclass.pred = multi_pred_classes
  )

# Save predictions
write.csv(pred_df, "results/rnn_predictions.csv", row.names = FALSE)

# Display the predictions
print(head(pred_df))


