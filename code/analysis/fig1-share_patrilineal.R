library(tidyverse)

df_community <- read_csv(here::here("datacloud","processed", "baseline.csv"))

df_community %>%
  group_by(round) %>%
  dplyr::summarize(egal_share = mean(egalitarian)) %>%
  mutate(patri_share = 1 - egal_share) %>%
  ggplot(aes(x = round, y = patri_share)) +
  expand_limits(y = c(0, 0.5)) +
  #scale_y_continuous(breaks = seq(0, 1, 0.1)) +
  # make y scale percent
scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  #scale_x_continuous(breaks = seq(0, num_round, num_cohort)) +
  geom_line() +
  theme_classic() +
  # only horizontal grid lines
  theme(panel.grid.major.y = element_line(color = "grey80"),
        panel.grid.minor.y = element_line(color = "grey90")) +
    labs(y = "Share of Unilineal Communities",
         x = "Round")

ggsave("figures/share_patrilineal.png", width = 6, height = 5)

max_round <- max(df_community$round)

df_inst_change_rank <- df_community %>%
  filter(round == max_round) %>%
  group_by(community_rank) %>%
  dplyr::summarize(egal_share = mean(egalitarian)) %>%
  mutate(patri_share = 1 - egal_share)  %>% 
  #turn to percent
  mutate(patri_share = scales::percent(patri_share, accuracy = 0.1)) %>% 
  select(community_rank, patri_share)  %>% 
  spread(community_rank, patri_share) 

# save table in figures
write_csv(df_inst_change_rank, here::here("figures", "share_patrilineal_table.csv"))


