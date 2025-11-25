# Install required R packages for tidymodels and XGBoost
# This script runs automatically on Streamlit Cloud

packages_to_install <- c("tidymodels", "tune", "xgboost", "stats")

for (pkg in packages_to_install) {
  if (!require(pkg, character.only = TRUE)) {
    cat(sprintf("Installing %s...\n", pkg))
    install.packages(pkg, repos = "https://cran.r-project.org/", quiet = TRUE)
  } else {
    cat(sprintf("%s already installed\n", pkg))
  }
}

cat("All R packages installed successfully!\n")
