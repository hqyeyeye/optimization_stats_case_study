pacman::p_load(dplyr, ggplot2, data.table, tidyr, ggthemes, viridis, showtext)
#font_import()
showtext_auto()

d = data.table::fread("data/lk_risks.csv") %>% 
  select(starts_with("risk_")) %>% 
  mutate_all(function(x)dense_rank(desc(x))) %>% 
  arrange(risk_logit) %>% 
  mutate(`route ID` = row_number()) %>% 
  pivot_longer(cols = risk_logit:risk_DL, 
               names_to = "stats model", 
               values_to = "risk rank") %>% 
  mutate(`stats model` = gsub("risk_", "", `stats model`)) %>% 
  mutate(`stats model` = case_when(`stats model` == "logit" ~ "Logistic",
                                   `stats model` == "poisson" ~ "Poisson",
                                   `stats model` == "xgboost" ~ "XGBoost",
                                   `stats model` == "DL" ~ "Deep Learning") %>% 
           factor(levels = c("Logistic", "Poisson", "XGBoost", "Deep Learning")))

p = d %>% 
  ggplot(aes(x = `route ID`, y = `stats model`, fill = `risk rank`)) + 
  scale_fill_viridis(option="magma", 
                     breaks = c(1, 5, 10, 15, 20), 
                     guide = guide_legend(reverse = FALSE)) + 
  scale_x_continuous(expand = c(0, 0), limits = c(0, 21)) + 
  geom_raster() + 
  theme_tufte(ticks = F) + 
  theme(plot.margin = unit(c(0, 0, 0, 0), "cm"),
        axis.title.y = element_blank())
#title = "Concordance of four models for evaluating the risk of crash",
       #caption = "Darker color indicates higher rank of crash risk") 
p

ggsave("Figures/concordance_4_models.pdf", p, width = 5, height = 2)
ggsave("Figures/concordance_4_models.png", p, width = 10*0.8, height = 4*0.8, dpi = 400)
ggsave("Figures/concordance_4_models.svg", p, width = 10, height = 4)
ggsave("Figures/concordance_4_models.eps", p, width = 10, height = 4)
