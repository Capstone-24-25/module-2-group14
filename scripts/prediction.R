require(tidyverse)
require(keras)
require(tensorflow)
load('data/claims-test.RData')
load('data/claims-raw.RData')
source('scripts/preprocessing.R')
options(askYesNo = function(...) TRUE)

# apply preprocessing pipeline
clean_df <- claims_test %>%
  slice(1:100) %>%
  parse_data() %>%
  select(.id, text_clean)

# grab input
x <- clean_df %>%
  pull(text_clean)

# compute predictions
preds_b <- predict(binary_model, x) %>%
  as.numeric()

preds_m <- predict(multi_model, x) %>%
  as.numeric()

class_labels_b <- claims_raw %>% pull(bclass) %>% levels()
class_labels_m <- claims_raw %>% pull(mclass) %>% levels()

pred_classes_b <- factor(preds_b > 0.5, labels = class_labels_b)
pred_classes_m <- factor(preds_m > 0.5, labels = class_labels_m)


# export (KEEP THIS FORMAT IDENTICAL)
pred_df <- clean_df %>%
  bind_cols(bclass.pred = pred_classes_b) %>%
  bind_cols(mclass.pred = pred_classes_m) %>%
  select(.id, bclass.pred)

save(pred_df, file = 'results/preds_group14.RData')

