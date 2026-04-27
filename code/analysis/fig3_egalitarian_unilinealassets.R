library(tidyverse)
library(patchwork)

df_baseline <- read_csv(here::here("datacloud","processed", "baseline.csv"))

plot_baseline <- df_baseline  %>% 
    mutate(community_rank = as.factor(community_rank)) %>%
    group_by(round, community_rank) %>%
    summarise(n_residents = mean(n_residents))  %>% 
    mutate(community_label = if_else(round == max(df_baseline$round), community_rank, "")) %>%
    ggplot(aes(x = round, y = n_residents, color = as.factor(community_rank))) +
    geom_line() +
    geom_text(aes(label = community_label), hjust = 1.1, vjust = -0.5, size = 3.5) +
    scale_y_continuous(breaks = seq(80, 250, 20)) +
    theme_classic()+
    # remove legend
    theme(
        legend.position = "none",
    # center title
        plot.title = element_text(hjust = 0.5))+
    # black and while color scale
    scale_color_grey() +
    labs(y = "Number of Residents",
         x = "Round",
         title = "Egalitarian Private Assets Inheritance") +
    expand_limits(y = c(80, 220))

plot_baseline

df_assetpatri <- read_csv(here::here("datacloud","processed", "assetpatri.csv"))

plot_assetpatri <- df_assetpatri  %>% 
    mutate(community_rank = as.factor(community_rank)) %>%
    group_by(round, community_rank) %>%
    summarise(n_residents = mean(n_residents))  %>% 
    mutate(community_label = if_else(round == max(df_assetpatri$round), community_rank, "")) %>%
    ggplot(aes(x = round, y = n_residents, color = as.factor(community_rank))) +
    geom_line() +
    geom_text(aes(label = community_label), hjust = 1.1, vjust = -0.2, size = 3.5) +
    scale_y_continuous(breaks = seq(80, 250, 20)) +
    theme_classic()+
    # remove legend
    theme(
        legend.position = "none",
        plot.title = element_text(hjust = 0.5))+
    # black and while color scale
    scale_color_grey() +
    labs(y = "Number of Residents",
         x = "Round",
         title = "Unilineal Private Assets Inheritance") +
    expand_limits(y = c(80, 220))

plot_assetpatri


plot_combined <- plot_baseline + plot_assetpatri

plot_combined

ggsave("figures/egalitarian_unilinealassets.png", plot = plot_combined, width = 7, height = 3.5)
