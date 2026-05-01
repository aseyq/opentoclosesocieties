# run_all_analysis.R

# Set the directory containing the scripts
analysis_dir <- file.path(here::here("code", "analysis"))
  

# List all .R files in the directory
r_files <- list.files(path = analysis_dir, pattern = "\\.R$", full.names = TRUE)

# Sort the files to run in order (if needed)
r_files <- sort(r_files)

# Run each script
for (file in r_files) {
  message("Running: ", file)
  source(file)
}
