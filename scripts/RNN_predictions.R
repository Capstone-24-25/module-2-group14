
### Claims-test Data

library(keras)
library(tensorflow)
library(tidyverse)
load('data/claims-test.RData')
load('data/claims-raw.RData')
source('scripts/preprocessing.R')

# Load trained RNN models
binary_model <- load_model_tf("results/binary-model")
multi_model <- load_model_tf("results/multi-model")

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
binary_preds <- binary_model %>%
  predict(test_sequences) %>%
  round()  # Convert probabilities to 0/1 predictions

# Generate multi-class predictions
multi_preds <- multi_model %>%
  predict(test_sequences) %>%
  k_argmax()  # Convert probabilities to class indices

# Map predictions to class labels
class_labels_binary <- c("Negative", "Positive")  # Adjust based on your binary labels
class_labels_multi <- levels(factor(claims_clean$mclass))  # Multi-class labels from training

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
