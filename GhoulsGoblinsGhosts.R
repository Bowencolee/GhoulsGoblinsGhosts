##
## Ghouls, Goblins, and Ghosts
##

library(tidymodels)
library(vroom)


### DATA ###
# setwd("C:/Users/bowen/Desktop/Stat348/GhoulsGoblinsGhosts")
ggg_train <- vroom::vroom("train.csv")%>%
  mutate(type = as.factor(type),
         color = as.factor(color))
ggg_test <- vroom::vroom("test.csv") %>%
  mutate(color = as.factor(color))


##### Recipe making #####

my_recipe <- recipe(type~., data=ggg_train) %>%
  update_role(id, new_role = "id") %>%
  step_zv(all_numeric_predictors())


prepped_recipe <- prep(my_recipe)
baked_recipe <- bake(prepped_recipe, ggg_train)

# rmse_vec(ggg_train[is.na(ggg_NA)], baked_recipe[is.na(ggg_NA)]) ## for missing values

##### Models We Know #####
 svmRadial_model <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>% # set or tune
               set_mode("classification") %>%
               set_engine("kernlab")
 
 svm_wf <- workflow() %>%
   add_recipe(my_recipe) %>%
   add_model(svmRadial_model)
 
 tuning_grid <- grid_regular(rbf_sigma(),
                             cost(),
                             levels = 7)
 
 folds <- vfold_cv(ggg_train, v = 7, repeats = 1)
 
 CV_results <- svm_wf %>%
   tune_grid(resamples=folds,
             grid=tuning_grid,
             metrics=metric_set(accuracy)) #f_meas,sens, recall,spec, precision, accuracy
 
 bestTune <- CV_results %>%
   select_best("accuracy")
 
 final_wf <- svm_wf %>%
   finalize_workflow(bestTune) %>%
   fit(data=ggg_train)
 
 svm_preds <- predict(final_wf, new_data=ggg_test,type="class") %>%
   bind_cols(., ggg_test) %>%
   select(id,.pred_class) %>%
   rename(type=.pred_class)
 
 vroom_write(x=svm_preds, file="./ggg_SVM.csv", delim=",")
