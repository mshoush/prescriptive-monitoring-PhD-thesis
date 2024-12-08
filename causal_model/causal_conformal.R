# Load necessary libraries
library(cfcausal)
library(arrow)

# Suppress warnings
options(warn = -1)

# Print command line arguments (if any)
args <- commandArgs(trailingOnly = TRUE)
print(args)


# Files paths
file_path_train <- "/home/mshoush/5th/causal_results/classification/bpic2012/updated_bpic2012_train.parquet"
file_path_test <- "/home/mshoush/5th/causal_results/classification/bpic2012/updated_bpic2012_test.parquet"
file_path_val <- "/home/mshoush/5th/causal_results/classification/bpic2012/updated_bpic2012_val.parquet"


# for VM

# # Check for dataset name argument
# if (length(args) < 1) {
#   stop("No dataset name provided. Please specify the dataset name as the first argument.")
# }
# dataset_name <- args[1]

# # Construct file paths dynamically based on dataset name
# file_path_train <- paste0("/home/centos/phd/5th/causal_results/classification/", dataset_name, "/updated_", dataset_name, "_train.parquet")
# file_path_test <- paste0("/home/centos/phd/5th/causal_results/classification/", dataset_name, "/updated_", dataset_name, "_test.parquet")
# file_path_val <- paste0("/home/centos/phd/5th/causal_results/classification/", dataset_name, "/updated_", dataset_name, "_val.parquet")


# Error handling for reading Parquet files
tryCatch({
  # Read the Parquet files into DataFrames
  train <- read_parquet(file_path_train)
  test <- read_parquet(file_path_test)
  val <- read_parquet(file_path_val)
}, error = function(e) {
  cat("Error loading Parquet files:\n", e$message, "\n")
  stop("Exiting script due to error in loading data.")
})

# Display the number of rows and columns in each DataFrame
cat("Train Data - Rows:", nrow(train), "Columns:", ncol(train), "\n")
cat("Test Data - Rows:", nrow(test), "Columns:", ncol(test), "\n")
cat("Validation Data - Rows:", nrow(val), "Columns:", ncol(val), "\n")


# Ensure the data has necessary columns for conformalIte
required_cols <- c("Treatment", "label")
if (!all(required_cols %in% colnames(train))) {
  cat("Missing required columns in train data.\n")
  stop("Missing required columns in train data.")
}
if (!all(required_cols %in% colnames(test))) {
  cat("Missing required columns in test data.\n")
  stop("Missing required columns in test data.")
}
if (!all(required_cols %in% colnames(val))) {
  cat("Missing required columns in validation data.\n")
  stop("Missing required columns in validation data.")
}


# Define the function to preprocess data
preprocess_data <- function(data) {
  # Extract treatment, outcome, and features
  T <- as.numeric(data$Treatment)  # Correct column name for treatment
  probs_0 <- data$probs_0
  probs_1 <- data$probs_1
  Y <- ifelse(T == 1, probs_1, probs_0)
  X <- data[, !(names(data) %in% c("Treatment", "label", "probs_0", "probs_1"))]
  
  return(list(X = as.matrix(X), T = T, Y = Y))
}


# Preprocess the train, test, and validation data
train_data <- preprocess_data(train)
test_data <- preprocess_data(test)
val_data <- preprocess_data(val)

# Define the function to calculate conformal intervals using conformalIte
calculate_and_save_conformal_intervals <- function(train_data, test_data, val_data, alpha_values, output_dir) {
  # Create results directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Calculate conformal intervals for each alpha value
  for (alpha in alpha_values) {
    # Run conformalIte
    CI_model <- conformalIte(
      X = train_data$X, 
      Y = train_data$Y, 
      T = train_data$T, 
      alpha = alpha,
      algo = "counterfactual", # counterfactual
      type = "CQR",
      quantiles = c(0.05, 0.95),
      outfun = "quantRF",
      useCV = FALSE
    )
    cat("alpha: ", alpha, "\n")
    # Calculate intervals for test data
    cat("Calculating intervals for test data...\n")
    test_intervals <- CI_model(test_data$X, test_data$Y, test_data$T)
    # test_intervals <- CI_model(test_data$X)
    test_data[paste0("lower_counterfactual_alpha_", alpha)] <- test_intervals$lower
    test_data[paste0("upper_counterfactual_alpha_", alpha)] <- test_intervals$upper

    # Save results to CSV
    write.csv(test_data, file = file.path(output_dir, paste0("conformal_intervals_test_alpha_", alpha, ".csv")), row.names = FALSE)
    
    # Calculate intervals for validation data
    cat("Calculating intervals for validation data...\n")
    val_intervals <- CI_model(val_data$X, val_data$Y, val_data$T)
    # val_intervals <- CI_model(val_data$X)
    val_data[paste0("lower_counterfactual_alpha_", alpha)] <- val_intervals$lower
    val_data[paste0("upper_counterfactual_alpha_", alpha)] <- val_intervals$upper
    
    # Save results to CSV
    #write.csv(test_data, file = file.path(output_dir, paste0("conformal_intervals_test_alpha_", alpha, ".csv")), row.names = FALSE)
    write.csv(val_data, file = file.path(output_dir, paste0("conformal_intervals_val_alpha_", alpha, ".csv")), row.names = FALSE)
  }
}
dataset_name <- "bpic2012"

# Define alpha values and output directory
alpha_values <- seq(0.1, 1.0, by = 0.1)
output_dir <- paste0("/home/mshoush/5th/conformal_causal_results/", dataset_name)
# for VM
# output_dir <- paste0("/home/centos/phd/5th/conformal_causal_results/", dataset_name)

# Calculate and save conformal intervals
calculate_and_save_conformal_intervals(train_data, test_data, val_data, alpha_values, output_dir)

cat("Conformal intervals calculated and saved.\n")




