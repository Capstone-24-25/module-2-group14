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
library(tidyverse)
library(tidytext)
library(stopwords)
library(randomForest)

# Load preprocessed data
load('data/claims-clean-example.RData')
load('data/claims-test.RData')

# Combine train and test data for consistent feature extraction
train_data <- claims_clean
test_data <- claims_test

# Add identifier for the data source
train_data <- train_data %>% mutate(set = "train")
test_data <- test_data %>% mutate(set = "test")

# Combine datasets
combined_data <- bind_rows(train_data, test_data)

# Tokenization and TF-IDF
tfidf <- combined_data %>%
  unnest_tokens(word, text_clean) %>%
  anti_join(stop_words(language = "en"), by = "word") %>%  # Remove stopwords
  count(set, .id, word) %>%
  bind_tf_idf(word, .id, n) %>%
  pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0, names_repair = "unique")

# Split TF-IDF matrix back into train and test
train_matrix <- tfidf %>% filter(set == "train") %>% select(-set, -.id)
test_matrix <- tfidf %>% filter(set == "test") %>% select(-set, -.id)

# Extract labels
train_labels_binary <- train_data %>%
  pull(bclass) %>%
  as.factor()

train_labels_multi <- train_data %>%
  pull(mclass) %>%
  as.factor()

test_ids <- test_data$.id

# Train Random Forest models
# Binary Classification
binary_rf <- randomForest(
  x = as.data.frame(train_matrix),
  y = train_labels_binary,
  ntree = 100,
  importance = TRUE
)

# Multi-Class Classification
multi_rf <- randomForest(
  x = as.data.frame(train_matrix),
  y = train_labels_multi,
  ntree = 100,
  importance = TRUE
)

# Generate predictions
binary_preds <- predict(binary_rf, as.data.frame(test_matrix))
multi_preds <- predict(multi_rf, as.data.frame(test_matrix))

# Format predictions into a data frame
pred_df <- data.frame(
  .id = test_ids,
  bclass.pred = as.character(binary_preds),
  mclass.pred = as.character(multi_preds)
)

# Save models and predictions
saveRDS(binary_rf, "results/random_forest_binary_model.rds")
saveRDS(multi_rf, "results/random_forest_multi_model.rds")
write.csv(pred_df, "results/random_forest_predictions.csv", row.names = FALSE)

# Display results
print(head(pred_df))