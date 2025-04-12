# data set

library(tidyverse)
library(mlr3verse)

task <- tsk("penguins")
task
task$data() %>% head
task%>% view()

head(task)

task$data() %>% summary

task %>% autoplot

#define a decision tree learner

mlr_learners %>%
  as.data.table() %>%
  view()

#as.data.table(mlr_filters)......does not display a good table

learner_rpart <- lrn("classif.rpart")#1

#model evaluation k-fold cross validation
rsmp_cv <- rsmp("cv", folds = 10 )#2
set.seed(42)

rsmp_cv$instantiate(task = task)#3    #to connect our task to the model evaluation cycle

design <- benchmark_grid(task, learner_rpart, rsmp_cv)
bmr <- benchmark(design)
bmr$aggregate()

#::::::::::::::::::::: this is NOT exactly the same as benchmark::::::::::::::::::::::::
#rr = resample(task, learner, resampling) and rr <- benchmark(task, learner, resampling) are different
#resample() is for evaluating a single model on a single dataset.
#benchmark() is for comparing multiple models on one or more datasets.

rr <- resample(task,learner_rpart, rsmp_cv)#3
rr$aggregate() #4
#::::::::::::::::::::: this is NOT exactly the same as  benchmark::::::::::::::::::::::::


#...............C.................

learner_ranger <- lrn("classif.ranger")
design_ranger <- benchmark_grid(task, learner_ranger, rsmp_cv)
bmr_ranger <- benchmark(design_ranger)#error due to missing values in cols that need imputing
task$missings()

#::::::::::::::::::::: this is exactly the same as  benchmark:::::::::::::::::::

#model evaluation
#rsmp_cv <- rsmp("cv", folds = 10)
#set.seed(42) # no need for this since we already set it at the begining

#resampling_ranger$instantiate(task) # we already did this, no need to do it again connect our task to model
r_forest <- resample(task, learner_ranger, rsmp_cv) #we get an error we notice missing values in our 
#data hence we need to deal with them

#::::::::::::::::::::: this is exactly the same as  benchmark:::::::::::::::::::

#..........................D.................................

#we need to impute because of some missing column values in the task
task$missings()

#we need to use mlr3pipeline  operators
library(mlr3pipelines)

po()#view all pipeoperators

graph_ranger <-  
  po("imputemedian") %>>%
  po("imputemode") %>>%
  lrn("classif.ranger") # so in short we have created a new learner which incorporates
learner_ranger_gr <-  as_learner(graph_ranger)
design_ranger_gr <- benchmark_grid(task, learner_ranger_gr, rsmp_cv)
bmr_ranger <- benchmark(design_ranger_gr)
bmr_ranger$aggregate()


#::::::::::::::::::::: this is here exactly the same as  benchmark:::::::::::::::::::::::
rr_ranger <- resample(task,learner_ranger_gr, rsmp_cv)
rr_ranger$aggregate()
#::::::::::::::::::::: this is here exactly the same as  benchmark:::::::::::::::::::::::

#................................E........................................
learner_svm <- lrn("classif.svm")
design_svm <- benchmark_grid(task, learner_svm, rsmp_cv)
bmr_svm <- benchmark(design_svm) # error arises due to unsupported features factor

#:::::::::::::::::we have done this with model comparison benchmark::::
resampling_svm <- resample(task, learner_svm, resampling)
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::.:
graph_svm <-  
  po("imputemedian") %>>%
  po("imputemode") %>>%
  po("encode")%>>%
  lrn("classif.svm")
learner_svm_gr <-  as_learner(graph_svm) 
design_svm_gr <- benchmark_grid(task, learner_svm_gr, rsmp_cv)
bmr_svm <- benchmark(design_svm_gr) 
bmr_svm$aggregate()

#::::::::::::::::::::: this is exactly the same as  benchmark:::::::::::::::::::

rr_svm <- resample(task, learner_svm_gr,resampling)
rr_svm$aggregate()
#::::::::::::::::::::: this is exactly the same as  benchmark:::::::::::::::::::

#.............f............
learners <- list(
                 learner_rpart,
                 learner_ranger_gr,
                 learner_svm_gr,
                 lrn('classif.featureless')
                 )
#resampling <- rsmp('cv', folds = 10)
#set.seed(45) # to ensure the split 
benchmark_designs = benchmark_grid(task, learners, rsmp_cv)
bmr_designs <- benchmark(benchmark_designs)

bmr_designs %>% autoplot

#resampling_ranger$instantiate(task) #  connect our task to model
#r_forest <- resample(task, learner_ranger, resampling)

#:::::::::::::Question2 :::::::::::

task <- tsk("moneyball")
task%>%view

task$data()%>%head()%>%view()

task%>% autoplot

mlr_learners%>%as.data.table%>%view()

#create a suitable SVM learner
learner_svm <- lrn(
                    "regr.svm", 
                    type = "nu-regression", 
                    kernel = "radial"
                    )
??svm
#preprocess missing data using robustify

library(mlr3pipelines)

svm_graph <- ppl("robustify", task = task, learner = learner_svm) %>>% learner_svm
graph_learner <- as_learner(svm_graph)
#graph_learner$train(new_data)
set.seed(42)
rsmp_cv <- rsmp("cv", folds = 10)
rsmp_cv$instantiate(task = task) # this will ensure consistent splits between runs hence reproducibility

learners = list(graph_learner,lrn("regr.rpart"), lrn("regr.featureless"))
                
design = benchmark_grid (learner = learners,
                        task = task,
                        resamplings = rsmp_cv
                        ) 
bmr = benchmark(design)                     
bmr$aggregate ()             

#defining an auto tuner with svm
auto.svm <- AutoTuner$new(
                          learner = graph_learner,
                          resampling = rsmp("holdout"),
                          search_space = ps(
                            c=p_dbl(lower = 10 ** -5, upper = 10 ** 5, trafo = function(x) 10 ** x),
                            kernel = p_fct(levels = c("polydot", "rbfdot"))
                             ),# best method is to save this as a variable in R
                          measure = msr("regr.rmse"),
                          tuner = tnr("cmaes"),
                          terminator = trm("evals", n_evals= 100)
                                                    )
library(mlr3tuning)
library(paradox)
install.packages("adagio")
library(adagio)


#:::::: Question E::::::::::::::
#defining an auto tuner with random search tuner
auto.random <- AutoTuner$new(
  learner = graph_learner,
  resampling = rsmp("holdout"),
  search_space = ps(
    c=p_dbl(lower = 10 ** -5, upper = 10 ** 5, trafo = function(x) 10 ** x),
    kernel = p_fct(levels = c("polydot", "rbfdot"))
  ),
  measure = msr("regr.rmse"),
  tuner = tnr("random_search"),
  terminator = trm("evals", n_evals= 100)
)

auto.random$id <- "holdout_search_turner"
auto.random$id

#::::::::f::::::::::::::::
#evaluate results for all learners
learner_summ = list(graph_learner,
                    lrn("regr.rpart"), 
                    lrn("regr.featureless"), 
                    auto.random, 
                    auto.svm)

design_summ = benchmark_grid (learner = learner_summ,
                         task = task,
                         resamplings = rsmp_cv
) 
bmr_summ = benchmark(design_summ)                     

bmr_summ%>% autoplot            
