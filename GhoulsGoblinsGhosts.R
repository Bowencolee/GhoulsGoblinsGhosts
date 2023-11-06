##
## Ghouls, Goblins, and Ghosts
##

library(tidymodels)
library(vroom)
library(nnet) # neural networks


### DATA ###
# setwd("C:/Users/bowen/Desktop/Stat348/GhoulsGoblinsGhosts")
ggg_train <- vroom::vroom("train.csv")%>%
  mutate(type = as.factor(type),
         color = as.factor(color))
ggg_test <- vroom::vroom("test.csv") %>%
  mutate(color = as.factor(color))


##### Recipe making #####

my_recipe <- recipe(type~., data=ggg_train) %>%
  update_role(id, new_role="id") %>%
  step_dummy(color) %>% ## Turn color to factor then dummy encode color
  step_range(all_numeric_predictors(), min=0, max=1) #scale to [0,1]



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
##### Neural Networks #####

nn_model <- mlp(hidden_units = tune(),
                epochs = 250) %>% #or 100 or 250
          set_engine("nnet") %>% #verbose = 0 prints off less
          set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nn_model)

tuning_grid <- grid_regular(hidden_units(range=c(1, 200)),
                            levels=5)

folds <- vfold_cv(ggg_train, v = 5, repeats = 1)

CV_results <- nn_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy)) #f_meas,sens, recall,spec, precision, accuracy

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=ggg_train)

nn_preds <- predict(final_wf, new_data=ggg_test,type="class") %>%
  bind_cols(., ggg_test) %>%
  select(id,.pred_class) %>%
  rename(type=.pred_class)

vroom_write(x=nn_preds, file="./ggg_NN.csv", delim=",")

  
CV_results %>% collect_metrics() %>%
  filter(.metric=="accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()  
