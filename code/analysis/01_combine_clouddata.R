library(tidyverse)

# Set directories
raw_dir <- here::here("datacloud", "raw")
processed_dir <- here::here("datacloud", "processed")

# Create processed directory if it doesn't exist
if (!dir.exists(processed_dir)) {
  dir.create(processed_dir, recursive = TRUE)
}

# List all subfolders in raw_dir
subfolders <- list.dirs(raw_dir, recursive = FALSE, full.names = TRUE)

# Iterate over each subfolder
for (subfolder in subfolders) {
  folder_name <- basename(subfolder)
  
  csv_files <- list.files(subfolder, pattern = "\\.csv$", full.names = TRUE)
  
  # Combine all CSVs using readr::read_csv and purrr::map_dfr
  combined_df <- map_dfr(csv_files, ~ read_csv(.x, show_col_types = FALSE) %>%
                           mutate(source_folder = folder_name))
  
  # Save the combined CSV
  output_path <- file.path(processed_dir, paste0(folder_name, ".csv"))
  write_csv(combined_df, output_path)
  
  message("Saved: ", output_path)
}

