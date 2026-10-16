#!/usr/bin/env Rscript

# Beginner-friendly statistical analysis for the cleaned heavy-equipment data.
# Only base R is used so the analysis is easy to install, read, and reproduce.

parse_arguments <- function(arguments) {
  values <- list(
    input = "data/processed/clean_train.csv",
    output_dir = "reports"
  )
  for (argument in arguments) {
    if (startsWith(argument, "--input=")) {
      values$input <- sub("^--input=", "", argument)
    } else if (startsWith(argument, "--output-dir=")) {
      values$output_dir <- sub("^--output-dir=", "", argument)
    } else {
      stop(paste("Unknown argument:", argument), call. = FALSE)
    }
  }
  values
}

safe_correlation <- function(left, right) {
  complete <- complete.cases(left, right)
  if (sum(complete) < 3 || sd(left[complete]) == 0 || sd(right[complete]) == 0) {
    return(NA_real_)
  }
  cor(left[complete], right[complete], method = "pearson")
}

required_columns <- c("TargetValue", "Age", "OperationalHoursMeter")
arguments <- parse_arguments(commandArgs(trailingOnly = TRUE))

if (!file.exists(arguments$input)) {
  stop(
    paste0(
      "Cleaned data not found at '", arguments$input,
      "'. Run 'python -m src.clean_data' first."
    ),
    call. = FALSE
  )
}

data <- read.csv(arguments$input, stringsAsFactors = FALSE, check.names = FALSE)
missing_columns <- setdiff(required_columns, names(data))
if (length(missing_columns) > 0) {
  stop(
    paste("Cleaned data is missing required column(s):", paste(missing_columns, collapse = ", ")),
    call. = FALSE
  )
}

dir.create(arguments$output_dir, recursive = TRUE, showWarnings = FALSE)

data$TargetValue <- as.numeric(data$TargetValue)
data$Age <- as.numeric(data$Age)
data$OperationalHoursMeter <- as.numeric(data$OperationalHoursMeter)
if (!"LogOperationalHours" %in% names(data)) {
  data$LogOperationalHours <- ifelse(
    data$OperationalHoursMeter > 0,
    log1p(data$OperationalHoursMeter),
    NA_real_
  )
}

summary_rows <- data.frame(
  metric = c(
    "row_count",
    "mean_price",
    "median_price",
    "price_standard_deviation",
    "missing_age_count",
    "missing_operational_hours_count",
    "age_price_pearson_correlation",
    "operational_hours_price_pearson_correlation"
  ),
  value = c(
    nrow(data),
    mean(data$TargetValue, na.rm = TRUE),
    median(data$TargetValue, na.rm = TRUE),
    sd(data$TargetValue, na.rm = TRUE),
    sum(is.na(data$Age)),
    sum(is.na(data$OperationalHoursMeter)),
    safe_correlation(data$Age, data$TargetValue),
    safe_correlation(data$OperationalHoursMeter, data$TargetValue)
  ),
  interpretation = c(
    "Number of cleaned auction records analysed",
    "Average observed selling price",
    "Middle observed selling price; less sensitive to extreme values",
    "Spread of observed selling prices",
    "Rows without a valid calculated equipment age",
    "Rows without a valid operating-hours value",
    "Linear association only; this does not prove that age causes price changes",
    "Linear association only; this does not prove that usage causes price changes"
  ),
  stringsAsFactors = FALSE
)

write.csv(
  summary_rows,
  file.path(arguments$output_dir, "r_statistical_summary.csv"),
  row.names = FALSE
)

group_summaries <- list()
group_candidates <- c(
  "InventoryGroupDescription", "InventoryGroupCategory", "Spec_BaseClass",
  "RegionCode", "UtilizationTier"
)
for (column in group_candidates) {
  if (column %in% names(data)) {
    groups <- as.character(data[[column]])
    groups[is.na(groups) | groups == ""] <- "Unknown"
    grouped <- aggregate(
      data$TargetValue,
      by = list(group = groups),
      FUN = function(values) c(
        transaction_count = length(values),
        mean_price = mean(values, na.rm = TRUE),
        median_price = median(values, na.rm = TRUE)
      )
    )
    expanded <- data.frame(
      group_type = column,
      group = grouped$group,
      transaction_count = grouped$x[, "transaction_count"],
      mean_price = grouped$x[, "mean_price"],
      median_price = grouped$x[, "median_price"],
      stringsAsFactors = FALSE
    )
    group_summaries[[length(group_summaries) + 1]] <- expanded
  }
}

if (length(group_summaries) > 0) {
  group_summary <- do.call(rbind, group_summaries)
} else {
  group_summary <- data.frame(
    group_type = character(), group = character(), transaction_count = integer(),
    mean_price = numeric(), median_price = numeric()
  )
}
write.csv(
  group_summary,
  file.path(arguments$output_dir, "r_group_summary.csv"),
  row.names = FALSE
)

category_column <- if ("InventoryGroupDescription" %in% names(data)) {
  "InventoryGroupDescription"
} else if ("InventoryGroupCategory" %in% names(data)) {
  "InventoryGroupCategory"
} else if ("Spec_BaseClass" %in% names(data)) {
  "Spec_BaseClass"
} else {
  NA_character_
}

png(file.path(arguments$output_dir, "price_by_category.png"), width = 1200, height = 750, res = 120)
if (!is.na(category_column)) {
  category <- as.character(data[[category_column]])
  category[is.na(category) | category == ""] <- "Unknown"
  category_medians <- sort(tapply(data$TargetValue, category, median, na.rm = TRUE), decreasing = TRUE)
  category_medians <- head(category_medians, 10)
  par(mar = c(10, 5, 3, 1))
  barplot(
    category_medians,
    las = 2,
    col = "#167C80",
    border = NA,
    main = "Median selling price by equipment category",
    ylab = "Median TargetValue"
  )
} else {
  plot.new()
  text(0.5, 0.5, "No equipment category column is available")
}
dev.off()

png(file.path(arguments$output_dir, "age_vs_price.png"), width = 1100, height = 750, res = 120)
valid_age <- complete.cases(data$Age, data$TargetValue)
if (sum(valid_age) >= 3) {
  valid_indices <- which(valid_age)
  set.seed(42)
  plot_indices <- if (length(valid_indices) > 5000) {
    sample(valid_indices, 5000)
  } else {
    valid_indices
  }
  plot(
    data$Age[plot_indices], data$TargetValue[plot_indices],
    pch = 19, col = rgb(22, 124, 128, 90, maxColorValue = 255),
    xlab = "Equipment age at transaction", ylab = "TargetValue",
    main = "Equipment age and observed selling price"
  )
  trend <- lm(log1p(TargetValue) ~ Age, data = data[valid_age, ])
  age_grid <- data.frame(
    Age = seq(min(data$Age[valid_age]), max(data$Age[valid_age]), length.out = 200)
  )
  trend_price <- pmax(0, expm1(predict(trend, newdata = age_grid)))
  lines(age_grid$Age, trend_price, col = "#D97706", lwd = 2)
} else {
  plot.new()
  text(0.5, 0.5, "Not enough valid age and price observations")
}
dev.off()

model_data <- data[complete.cases(data[, c("TargetValue", "Age", "LogOperationalHours")]), ]
png(file.path(arguments$output_dir, "regression_diagnostics.png"), width = 1200, height = 900, res = 120)
if (nrow(model_data) >= 8 && length(unique(model_data$TargetValue)) >= 3) {
  price_model <- lm(log1p(TargetValue) ~ Age + LogOperationalHours, data = model_data)
  par(mfrow = c(2, 2))
  plot(price_model)
  model_summary <- summary(price_model)
  coefficient_rows <- data.frame(
    metric = paste0("regression_coefficient_", rownames(model_summary$coefficients)),
    value = model_summary$coefficients[, "Estimate"],
    interpretation = "Estimated association in a log-price regression; not a causal effect",
    stringsAsFactors = FALSE
  )
  model_rows <- data.frame(
    metric = c("regression_r_squared", "regression_adjusted_r_squared"),
    value = c(model_summary$r.squared, model_summary$adj.r.squared),
    interpretation = c(
      "Fraction of in-sample log-price variation explained by the simple model",
      "R-squared adjusted for the number of predictors"
    ),
    stringsAsFactors = FALSE
  )
  write.table(
    rbind(coefficient_rows, model_rows),
    file.path(arguments$output_dir, "r_statistical_summary.csv"),
    sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE
  )
} else {
  plot.new()
  text(0.5, 0.5, "Not enough complete rows for regression diagnostics")
}
dev.off()

cat("R analysis complete:", arguments$output_dir, "\n")
cat("Reminder: association does not establish causation.\n")
