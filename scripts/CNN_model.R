# Load required libraries
library(tidyverse)
library(keras)
library(tensorflow)
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
test_ids <- testing(partitions) %>%
  pull(.id)

# Define vocabulary size and sequence length
vocab_size <- 10000
sequence_length <- 100

# Create a text vectorization layer
vectorize_layer <- layer_text_vectorization(
  max_tokens = vocab_size,
  output_sequence_length = sequence_length
)

# Adapt the layer on training text
vectorize_layer %>% adapt(train_text)

# Vectorize the text data
train_sequences <- vectorize_layer(train_text)
test_sequences <- vectorize_layer(test_text)

# Define CNN model architecture for binary classification
binary_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, input_length = sequence_length) %>%
  layer_conv_1d(filters = 128, kernel_size = 5, activation = 'relu') %>%
  layer_global_average_pooling_1d() %>%  # Reduce sequence to a single value
  layer_dropout(0.5) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 1, activation = 'sigmoid') 

# Compile the binary model
binary_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = 'binary_accuracy'
)

# Train the binary model
binary_history <- binary_model %>% fit(
  x = train_sequences,
  y = train_labels_binary,
  validation_split = 0.2,
  epochs = 10,
  batch_size = 32
)

# Define CNN model architecture for multi-class classification
num_classes <- length(unique(train_labels_multi))

multi_model <- keras_model_sequential() %>%
  layer_embedding(input_dim = vocab_size, output_dim = 128, input_length = sequence_length) %>%
  layer_conv_1d(filters = 128, kernel_size = 5, activation = 'relu') %>%
  layer_global_average_pooling_1d() %>%  # Reduce sequence to a single value
  layer_dropout(0.5) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Compile the multi-class model
multi_model %>% compile(
  loss = 'sparse_categorical_crossentropy',
  optimizer = 'adam',
  metrics = 'sparse_categorical_accuracy'
)

# Train the multi-class model
multi_history <- multi_model %>% fit(
  x = train_sequences,
  y = train_labels_multi,
  validation_split = 0.2,
  epochs = 10,
  batch_size = 32
)

# Save the models
save_model_tf(binary_model, "results/cnn_binary_model")
save_model_tf(multi_model, "results/cnn_multi_model")

# Generate predictions
# Convert predictions to numeric
binary_preds <- as.numeric(binary_preds)  # Ensure binary_preds is numeric
multi_preds <- as.numeric(multi_preds)  # Ensure multi_preds is numeric

# Create the predictions data frame
pred_df <- data.frame(
  .id = test_ids,
  bclass.pred = ifelse(binary_preds == 1, "Positive", "Negative"),
  mclass.pred = levels(factor(claims_clean$mclass))[multi_preds + 1]
)

# Save predictions
write.csv(pred_df, "results/cnn_predictions.csv", row.names = FALSE)

# Display results
print(head(pred_df))