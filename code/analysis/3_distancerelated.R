library(tidyverse)

df_community <- read_csv(here::here("data","processed", "baseline.csv"))

max_round <- max(df_community$round)


compute_distances <- function(df) {
  coords <- df %>% select(location_x, location_y)
  dist_mat <- as.matrix(dist(coords))
  
  # Average distance to others
  avg_dist <- rowMeans(dist_mat)
  
  # Nearest neighbor distance: set self-distance to Inf to ignore
  dist_mat_no_diag <- dist_mat
  diag(dist_mat_no_diag) <- Inf
  
  nearest_dist <- apply(dist_mat_no_diag, 1, min)
  
  tibble(
    community_name = df$community_name,
    avg_pairwise_distance = avg_dist,
    nearest_neighbor_distance = nearest_dist
  )
}

df_distance_measures <- df_community  %>% 
  filter(round == max_round) %>% 
  select(sim, community_name, location_x, location_y) %>%
  group_by(sim) %>%
  group_modify(~ compute_distances(.x)) %>%
  ungroup()  


df_community2 <- df_community %>%
  left_join(df_distance_measures, by = c("sim", "community_name"))


# Define breaks with Inf to catch any high values
breaks <- c(seq(0.6, 2.8, by = 0.4), Inf)

# Create custom labels, last one will be "2.6+"
labels <- c(
  paste0(head(breaks, -2), "–", head(tail(breaks, -1), -1)),
  "2.6+"
)

df_binned <- df_community2 %>%
  mutate(
    pairwise_bin = cut(
      avg_pairwise_distance,
      breaks = breaks,
      include.lowest = TRUE,
      right = FALSE,
      labels = labels
    )
  ) %>%
  group_by(pairwise_bin, community_rank) %>%
  summarise(
    patrilineal_rate = 1 - mean(egalitarian),
    .groups = "drop"
  ) %>%
  drop_na()

# Plot
ggplot(df_binned, aes(x = pairwise_bin, y = factor(community_rank), fill = patrilineal_rate)) +
  geom_tile(color = "white") +
  scale_y_discrete(limits = rev) +
  scale_fill_viridis_c(
    name = "Unilineal \n Prop.",
    limits = c(0, 1),
    guide = guide_colorbar(
      barheight = 5,
      barwidth = 0.5
    )
  ) +
  labs(
    x = "Community Centrality (Avg. Pairwise Distance)",
    y = "Community Wealth Rank"
  ) +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7)
    # legend.position = "bottom"
  )

# Save the plot
ggsave("figures/switch_distancerank.png", width = 5, height = 4, dpi = 300)


###====== Regression 


df_logit <- df_community2  %>% 
  filter(generation == max(generation))   %>% 
  filter(round == max(round))  %>% 
  select(egalitarian, community_wealth, avg_pairwise_distance, nearest_neighbor_distance, location_x, location_y)  %>% 
  mutate(egalitarian = as.numeric(egalitarian))  %>% 
  mutate(patrilineal = 1 - egalitarian)


##logit with 
# egalitarian ~ community_wealth + avg_pairwise_distance 
model <- glm(patrilineal ~ community_wealth + avg_pairwise_distance, 
           data = df_logit, family = binomial(link = "logit"))
##basic lm
model1 <- lm(patrilineal ~ community_wealth + avg_pairwise_distance, 
            data = df_logit)
model2 <- lm(patrilineal ~ community_wealth + nearest_neighbor_distance,
            data = df_logit)
model3 <- glm(patrilineal ~ community_wealth + avg_pairwise_distance, 
            data = df_logit, family = binomial(link = "logit"))
model4 <- glm(patrilineal ~ community_wealth + nearest_neighbor_distance, 
            data = df_logit, family = binomial(link = "logit"))
texreg::screenreg(list(model1, model2, model3, model4), single.row = TRUE)



library(texreg)
library(officer)
library(flextable)
library(htmlTable)

# model1 <- lm(egalitarian ~ community_wealth + avg_pairwise_distance, data = df_logit)
# model2 <- lm(egalitarian ~ community_wealth + nearest_neighbor_distance, data = df_logit)
model3 <- glm(egalitarian ~ community_wealth + avg_pairwise_distance, data = df_logit, family = binomial(link = "logit"))
model4 <- glm(egalitarian ~ community_wealth + nearest_neighbor_distance, data = df_logit, family = binomial(link = "logit"))

html_output <- capture.output(
  htmlreg(list(model1, model2), doctype = FALSE, single.row = TRUE)
)
# save the HTML output to a file
writeLines(html_output, "figures/model_summary.html")
