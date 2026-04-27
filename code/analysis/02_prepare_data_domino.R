library(tidyverse)


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


bind_rows(df_domino1, df_domino2, df_domino3, df_domino4, df_domino5, df_domino6, df_domino7) %>%
  group_by(round) %>%
  dplyr::summarize(egal_share = mean(egalitarian)) %>%
  mutate(patri_share = 1 - egal_share) %>%
  select(round, patri_share) %>%
  mutate(treatment = "Domino") -> df_patri_share