library(tidyverse)


## ==== Baseline =====
df_baseline <- read_csv(here::here("datacloud","processed", "baseline.csv"))

df_baseline_share <- df_baseline %>%
  group_by(round) %>%
  dplyr::summarize(egal_share = mean(egalitarian)) %>%
  mutate(patri_share = 1 - egal_share)  %>% 
  select(round, patri_share) %>%
  mutate(treatment = "Baseline") 


## ==== Domino =====
df_domino1<- read_csv(here::here("datacloud","processed", "domino1.csv"))  %>% 
    filter(community_rank == 1)

df_domino2 <- read_csv(here::here("datacloud","processed", "domino2.csv"))  %>% 
    filter(community_rank == 2)

df_domino3 <- read_csv(here::here("datacloud","processed", "domino3.csv"))  %>% 
    filter(community_rank == 3)

df_domino4 <- read_csv(here::here("datacloud","processed", "domino4.csv"))  %>% 
    filter(community_rank == 4)

df_domino5 <- read_csv(here::here("datacloud","processed", "domino5.csv"))  %>% 
    filter(community_rank == 5)

df_domino6 <- read_csv(here::here("datacloud","processed", "domino6.csv"))  %>% 
    filter(community_rank == 6)

df_domino7 <- read_csv(here::here("datacloud","processed", "domino7.csv"))  %>% 
    filter(community_rank == 7)


df_domino_share <- bind_rows(df_domino1, df_domino2, df_domino3, df_domino4, df_domino5, df_domino6, df_domino7) %>%
  group_by(round) %>%
  dplyr::summarize(egal_share = mean(egalitarian)) %>%
  mutate(patri_share = 1 - egal_share) %>%
  select(round, patri_share) %>%
  mutate(treatment = "w/o Domino effect") 


## ==== Lock-in =====
df_lockin1 <- read_csv(here::here("datacloud","processed", "lockin1.csv"))  %>% 
    filter(community_rank == 1)

df_lockin2 <- read_csv(here::here("datacloud","processed", "lockin2.csv"))  %>% 
    filter(community_rank == 2)

df_lockin3 <- read_csv(here::here("datacloud","processed", "lockin3.csv"))  %>% 
    filter(community_rank == 3)

df_lockin4 <- read_csv(here::here("datacloud","processed", "lockin4.csv"))  %>% 
    filter(community_rank == 4)

df_lockin5 <- read_csv(here::here("datacloud","processed", "lockin5.csv"))  %>% 
    filter(community_rank == 5)

df_lockin6 <- read_csv(here::here("datacloud","processed", "lockin6.csv"))  %>% 
    filter(community_rank == 6)

df_lockin7 <- read_csv(here::here("datacloud","processed", "lockin7.csv"))  %>% 
    filter(community_rank == 7)

df_lockin_share <- bind_rows(df_lockin1, df_lockin2, df_lockin3, df_lockin4, df_lockin5, df_lockin6, df_lockin7) %>%
  group_by(round) %>%
  dplyr::summarize(egal_share = mean(egalitarian)) %>%
  mutate(patri_share = 1 - egal_share) %>%
  select(round, patri_share) %>%
  mutate(treatment = "Lock-in effect") %>%
  ungroup()





bind_rows(df_baseline_share, df_domino_share, df_lockin_share) %>%
  mutate(nudge_y = case_when(
    treatment == "Baseline" ~ 0.025,
    treatment == "w/o Domino effect" ~ -0.02,
    treatment == "Lock-in effect" ~ 0.03
  )) %>%
  mutate(label = if_else(round == max(round-20), treatment, "")) %>%
  ggplot(aes(x = round, y = patri_share, linetype = treatment)) +
  expand_limits(y = c(0, 0.5)) +
  # make y scale percent
  scale_y_continuous(labels = scales::percent_format(accuracy = 1), breaks = seq(0, 1, by = 0.1)) +
  geom_line() +
    geom_text(aes(label = label, y = patri_share + nudge_y), size = 3.5, hjust = 0.5) +
  theme_classic() +
  # only horizontal grid lines
  theme(panel.grid.major.y = element_line(color = "grey80"),
        panel.grid.minor.y = element_line(color = "grey90")) +
    labs(y = "Share of Unilineal Communities",
         x = "Round") +
    #remove legend
    theme(legend.position = "none") 

ggsave("figures/share_baseline_lockin_domino.png", width = 6, height = 5)
