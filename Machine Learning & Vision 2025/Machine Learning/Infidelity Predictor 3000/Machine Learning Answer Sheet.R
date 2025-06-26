## Initial Setup----------------------------------------------------------------------------------------------------------------------------

#install.packages(c("tidyverse", "caret", "ipred", "AER","rpart"), dependencies = TRUE)

library(tidyverse)  
library(caret)
library(ipred)
library(rpart)
library(AER)

## Data Cleaning----------------------------------------------------------------------------------------------------------------------------

#Import from AER
data("Affairs")
# Create binary target: TRUE if they had one or more affairs
Affairs$cheater <- as.factor(Affairs$affairs > 0)
# Convert character vars to factors
Affairs$gender <- as.factor(Affairs$gender)
Affairs$children <- as.factor(Affairs$children)
# Drop 'rownames' and 'affairs' columns
affairs_clean <- Affairs %>%
  select( -affairs)

## Data Splitter----------------------------------------------------------------------------------------------------------------------------

set.seed(42)

split <- sample(1:nrow(affairs_clean), 0.7 * nrow(affairs_clean))
train <- affairs_clean[split, ]
test <- affairs_clean[-split, ]

#Setting cheating as a factor
train$cheater <- factor(train$cheater, levels = c("FALSE", "TRUE"))

# Upsampling Process
up_train <- upSample(x = train[, -which(names(train) == "cheater")], 
                     y = train$cheater)

names(up_train)[names(up_train) == "Class"] <- "cheater"


## Training the models----------------------------------------------------------------------------------------------------------------------------

# 1. Logistic LASSO Regression Training + Tuning ---------------------------------

set.seed(20346554)
lambdas <- 10^seq(-4, 1, length = 100)

mod.LASSO <- train(cheater ~ ., 
                   data = up_train, 
                   method = "glmnet", 
                   preProcess = NULL,
                   trControl = trainControl("repeatedcv", number = 10, repeats = 5,verboseIter = TRUE ),
                   tuneGrid = expand.grid(alpha = 1, lambda = lambdas)
)

bestLambda <- mod.LASSO$bestTune$lambda

# Tuned
mod.LASSO.posttune <- train(cheater ~ ., 
                            data = up_train,
                            method = "glmnet",
                            preProcess = c("center", "scale"),
                            trControl = trainControl(method = "none"),
                            tuneGrid = expand.grid(alpha = 1, 
                                                   lambda = bestLambda)
)


# 2. Bagging Trees Training + Tuning ---------------------------------------------

set.seed(20346554)

# Pretune bagging (basic)
btree.pretune <- bagging(cheater ~ ., data = up_train, nbagg = 25, coob = TRUE)

# Hyperparameter grid
grid <- expand.grid(nbagg = seq(25, 150, 25),
                    cp = seq(0, 0.5, 0.1),
                    minsplit = seq(5, 20, 5))

for (I in 1:nrow(grid)) {
  cat("Bagging Tuning:",I, " of",nrow(grid), "\n")
  set.seed(20346554)
  bagged.tree <- bagging(cheater ~ ., data = up_train,
                         nbagg = grid$nbagg[I],
                         coob = TRUE,
                         control = rpart.control(cp = grid$cp[I], minsplit = grid$minsplit[I]))
  
  grid$OOB.accuracy[I] <- 1 - bagged.tree$err
  pred <- predict(bagged.tree, newdata = up_train, type = "class")
  cm <- confusionMatrix(relevel(pred, ref = "TRUE"),
                        relevel(up_train$cheater, ref = "TRUE"))
  grid$test.accuracy[I] <- cm$overall["Accuracy"]
}

top5 <- grid[order(grid$OOB.accuracy, decreasing = TRUE)[1:5], ] %>%
  mutate(OOB.misclass.error = round(1 - OOB.accuracy, 4))

btree.bestTune <- top5 %>% slice(1)

# Final tuned Bagging
btree.tune <- bagging(cheater ~ ., data = up_train,
                      nbagg = btree.bestTune$nbagg,
                      coob = TRUE,
                      control = rpart.control(cp = btree.bestTune$cp,
                                              minsplit = btree.bestTune$minsplit))


# 3. Classification Trees Training + Tuning --------------------------------------

set.seed(20346554)

ctree.pretune <- rpart(cheater ~ ., data = up_train)

rtree.pr <- train(cheater ~ ., data = up_train, method = "rpart",
                  trControl = trainControl("cv", number = 10),
                  tuneLength = 15)

# Final tuned Classification Tree on full up_train data
ctree.tuned <- rpart(cheater ~ ., data = up_train, cp = rtree.pr$bestTune$cp)





## Menu Creation ---------------------------------------------------------------------------------------------------------------------------


# Helper to ask input with validation loop
ask_valid_input <- function(prompt, default = NULL, validate_fn, convert = identity) {
  repeat {
    cat(prompt)
    val <- readline()
    if (val == "" && !is.null(default)) {
      return(default)
    }
    # Try convert, catch errors
    safe_val <- tryCatch(convert(val), error = function(e) NA)
    if (!is.na(safe_val) && validate_fn(safe_val)) {
      return(safe_val)
    }
    cat("Invalid input. Please try again.\n")
  }
}


# Menu loop
repeat {
  choice <- menu(c("Predict cheating", "Exit"), title = "Infidelity Predictor 3000")
  
  if (choice == 1) {
    # --- User input ---
    
    gender_input <- ask_valid_input(
      "Gender (male/female): ", "female",
      validate_fn = function(x) tolower(x) %in% c("male", "female"),
      convert = tolower
    )
    
    age_input <- ask_valid_input(
      "Age (e.g. 32.0): ", 30,
      validate_fn = function(x) is.numeric(x) && (floor(x * 10) == x * 10),
      convert = as.numeric
    )
    
    yearsmarried_input <- ask_valid_input(
      "Years married (e.g. 7.000): ", 5,
      validate_fn = function(x) is.numeric(x) && (floor(x * 1000) == x * 1000),
      convert = as.numeric
    )
    
    children_input <- ask_valid_input(
      "Children (yes/no): ", "no",
      validate_fn = function(x) tolower(x) %in% c("yes", "no"),
      convert = tolower
    )
    
    religiousness_input <- ask_valid_input(
      "Religiousness (1–5): ", 3,
      validate_fn = function(x) is.numeric(x) && x %% 1 == 0 && x >= 1 && x <= 5,
      convert = as.integer
    )
    
    education_input <- ask_valid_input(
      "Education level (1–20): ", 16,
      validate_fn = function(x) is.numeric(x) && x %% 1 == 0 && x >= 1 && x <= 20,
      convert = as.integer
    )
    
    occupation_input <- ask_valid_input(
      "Occupation (1–7): ", 4,
      validate_fn = function(x) is.numeric(x) && x %% 1 == 0 && x >= 1 && x <= 7,
      convert = as.integer
    )
    
    rating_input <- ask_valid_input(
      "Marriage rating (1–5): ", 4,
      validate_fn = function(x) is.numeric(x) && x %% 1 == 0 && x >= 1 && x <= 5,
      convert = as.integer
    )
    
    # --- Build prediction frame ---
    new_data <- data.frame(
      gender = factor(gender_input, levels = levels(Affairs$gender)),
      age = age_input,
      yearsmarried = yearsmarried_input,
      children = factor(children_input, levels = levels(Affairs$children)),
      religiousness = religiousness_input,
      education = education_input,
      occupation = occupation_input,
      rating = rating_input
    )
    
    # ✨ Print dat mirror back to the user
    cat("\n-- You entered: --\n")
    print(new_data)
    cat("\n-- Predicting cheating behavior... --\n")
    
    # --- Place the Prediction here ---
    
    # Predict on user input with each tuned model 
    pred_lasso <- predict(mod.LASSO.posttune, newdata = new_data)
    pred_bagging <- predict(btree.tune, newdata = new_data, type = "class")
    pred_ctree <- predict(ctree.tuned, newdata = new_data, type = "class")
    
    # Map predictions to your custom strings
    lasso_str <- ifelse(as.character(pred_lasso) == "TRUE", "cheater!!!!", "Loyal as a dog")
    bagging_str <- ifelse(as.character(pred_bagging) == "TRUE", "Cooked....", " Would absolutely never")
    ctree_str <- ifelse(as.character(pred_ctree) == "TRUE", "would cheat", "Wont cheat")
    
    cat("\n===== 🔍 Prediction Results =====\n")
    cat("Predict LASSO : ", lasso_str, "\n")
    cat("Predict Bagging : ", bagging_str, "\n")
    cat("Predict Classification : ", ctree_str, "\n")
    cat("===============================\n")
    
  } else {
    cat("🗣️🗣️SEE YUH🗣️🗣️.\n")
    break
  }
}
   


