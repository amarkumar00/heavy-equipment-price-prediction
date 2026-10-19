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

# Formal hypothesis tests are reported separately from descriptive summaries.
# With a large dataset, tiny p-values are common, so each test also includes an
# effect-size estimate and a reminder that statistical significance is not
# proof of business importance or causation.
hypothesis_rows <- data.frame(
  test = character(),
  null_hypothesis = character(),
  p_value = numeric(),
  p_value_display = character(),
  effect_size_name = character(),
  effect_size = numeric(),
  interpretation = character(),
  stringsAsFactors = FALSE
)
minimum_hypothesis_group_size <- if (nrow(data) < 1000) 2 else 30

add_hypothesis <- function(
  test, null_hypothesis, p_value, effect_size_name, effect_size, interpretation
) {
  hypothesis_rows <<- rbind(
    hypothesis_rows,
    data.frame(
      test = test,
      null_hypothesis = null_hypothesis,
      p_value = p_value,
      p_value_display = format.pval(p_value, digits = 4, eps = 2.2e-16),
      effect_size_name = effect_size_name,
      effect_size = effect_size,
      interpretation = interpretation,
      stringsAsFactors = FALSE
    )
  )
}

if (!is.na(category_column)) {
  category_test_data <- data.frame(
    LogPrice = log1p(data$TargetValue),
    Category = as.factor(data[[category_column]])
  )
  category_test_data <- category_test_data[complete.cases(category_test_data), ]
  category_counts <- table(category_test_data$Category)
  retained_categories <- names(
    category_counts[category_counts >= minimum_hypothesis_group_size]
  )
  category_test_data <- category_test_data[
    category_test_data$Category %in% retained_categories,
  ]
  category_test_data$Category <- droplevels(category_test_data$Category)
  if (
    nrow(category_test_data) >= 2 * minimum_hypothesis_group_size &&
      nlevels(category_test_data$Category) >= 2
  ) {
    category_anova <- aov(LogPrice ~ Category, data = category_test_data)
    anova_table <- summary(category_anova)[[1]]
    category_p_value <- anova_table[["Pr(>F)"]][1]
    category_eta_squared <- anova_table[["Sum Sq"]][1] / sum(anova_table[["Sum Sq"]])
    add_hypothesis(
      "one_way_anova_log_price_by_category",
      "Mean log selling price is equal across retained equipment categories",
      category_p_value,
      "eta_squared",
      category_eta_squared,
      "Tests category differences; effect size is more useful than p-value alone"
    )
  }
}

if ("UtilizationTier" %in% names(data)) {
  utilization_test_data <- data.frame(
    LogPrice = log1p(data$TargetValue),
    UtilizationTier = as.character(data$UtilizationTier)
  )
  utilization_test_data <- utilization_test_data[
    complete.cases(utilization_test_data) &
      utilization_test_data$UtilizationTier %in% c("High", "Low"),
  ]
  high_prices <- utilization_test_data$LogPrice[
    utilization_test_data$UtilizationTier == "High"
  ]
  low_prices <- utilization_test_data$LogPrice[
    utilization_test_data$UtilizationTier == "Low"
  ]
  if (
    length(high_prices) >= minimum_hypothesis_group_size &&
      length(low_prices) >= minimum_hypothesis_group_size
  ) {
    utilization_test <- t.test(high_prices, low_prices, var.equal = FALSE)
    pooled_standard_deviation <- sqrt(
      ((length(high_prices) - 1) * var(high_prices) +
        (length(low_prices) - 1) * var(low_prices)) /
        (length(high_prices) + length(low_prices) - 2)
    )
    cohen_d <- if (pooled_standard_deviation > 0) {
      (mean(high_prices) - mean(low_prices)) / pooled_standard_deviation
    } else {
      NA_real_
    }
    add_hypothesis(
      "welch_t_test_high_vs_low_utilization_log_price",
      "High- and Low-utilization equipment have equal mean log selling price",
      utilization_test$p.value,
      "cohen_d",
      cohen_d,
      "Unadjusted group comparison; confounding can explain part of the difference"
    )
  }
}

write.csv(
  hypothesis_rows,
  file.path(arguments$output_dir, "r_hypothesis_tests.csv"),
  row.names = FALSE
)

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
cat("Formal hypothesis tests:", nrow(hypothesis_rows), "\n")
cat("Reminder: association does not establish causation.\n")
