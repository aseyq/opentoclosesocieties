library(tidyverse)

df_community <- read_csv(here::here("data","processed", "baseline.csv"))

library(ineq)
df_community %>%
  # filter(cohort==1) %>%
  mutate(wealth_pc_adjusted = if_else(egalitarian,
    community_wealth / n_residents,
    community_wealth / (2 * n_members)
  )) %>%
  group_by(round, sim) %>%
  dplyr::summarize(gini = ineq(wealth_pc_adjusted)) %>%
  group_by(round) %>%
  dplyr::summarize(gini = mean(gini)) %>%
  ggplot(aes(y = gini, x = round)) +
  geom_line() +
  scale_y_continuous(breaks = seq(0, 0.5, 0.1)) +
#   scale_x_continuous(breaks = seq(0, num_round, num_cohort)) +
  expand_limits(y = c(0, 0.5)) +
  theme_bw()

ggsave("figures/gini.png", width = 6, height = 4)

max_round <- max(df_community$round)

df_community %>%
  # filter(cohort==1) %>%
  mutate(wealth_pc_adjusted = if_else(egalitarian,
    community_wealth / n_residents,
    community_wealth / (2 * n_members)
  )) %>%
  group_by(round, sim) %>%
  dplyr::summarize(gini = ineq(wealth_pc_adjusted)) %>%
  group_by(round) %>%
  dplyr::summarize(gini = mean(gini))  %>%
  filter(round %in% c(0, max_round))  %>%
  write_csv(here::here("figures", "gini.csv"))
