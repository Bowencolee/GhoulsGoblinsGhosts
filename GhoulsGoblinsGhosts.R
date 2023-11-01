##
## Ghouls, Goblins, and Ghosts
##

library(tidymodels)
library(vroom)


### DATA ###
# setwd("C:/Users/bowen/Desktop/Stat348/GhoulsGoblinsGhosts")
ggg_NA <- vroom::vroom("trainWithMissingValues.csv")%>%
  mutate(type = as.factor(type))
ggg_train <- vroom::vroom("train.csv")%>%
  mutate(type = as.factor(type))
ggg_test <- vroom::vroom("test.csv")


##### Recipe making #####

my_recipe <- recipe(type~., data=ggg_NA) %>%
  step_impute_median(bone_length) %>%
  step_impute_median(hair_length) %>%
  step_impute_median(rotting_flesh)
  
  


prepped_recipe <- prep(my_recipe)
baked_recipe <- bake(prepped_recipe, ggg_NA)

rmse_vec(ggg_train[is.na(ggg_NA)], baked_recipe[is.na(ggg_NA)])
