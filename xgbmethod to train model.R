
# This script is loosely ported from Python script by modkzs, Misfyre and others
# https://www.kaggle.com/misfyre/allstate-claims-severity/encoding-feature-comb-modkzs-1108-72665
# However it gets a little better results maybe due to
# different realisations of scaling and Box Cox transformations in R and Python
#
#

library(data.table)
library(Matrix)
library(xgboost)
library(Metrics)




ID = 'id'
TARGET = 'loss'
SEED = 0
SHIFT = 200

TRAIN_FILE = "../train.csv"
TEST_FILE = "../test.csv"
SUBMISSION_FILE = "../input/sample_submission.csv"

# Read data

train = fread(TRAIN_FILE, showProgress = TRUE)
test = fread(TEST_FILE, showProgress = TRUE)

y_train = log(train[,TARGET, with = FALSE] + SHIFT)[[TARGET]]

train[, c(ID, TARGET) := NULL]
test[, c(ID) := NULL]

ntrain = nrow(train)
train_test = rbind(train, test)

features = names(train)

# Change character as factors 

for (f in features) {
        if (class(train_test[[f]])=="character") {
                #cat("VARIABLE : ",f,"\n")
                levels <- sort(unique(train_test[[f]]))
                train_test[[f]] <- as.integer(factor(train_test[[f]], levels=levels))
        }
}

# in order to speed up fit within Kaggle scripts have removed 30
# least important factors as identified from local run
features_to_drop <- c("cat67","cat21","cat60","cat65", "cat32", "cat30",
                      "cat24", "cat74", "cat85", "cat17", "cat14", "cat18",
                      "cat59", "cat22", "cat63", "cat56", "cat58", "cat55",
                      "cat33", "cat34", "cat46", "cat47", "cat48", "cat68",
                      "cat35", "cat20", "cat69", "cat70", "cat15", "cat62")

x_train = train_test[1:ntrain,-features_to_drop, with = FALSE]
x_test = train_test[(ntrain+1):nrow(train_test),-features_to_drop, with = FALSE]
# 
# Change data format in order to fit hte xgb packages

dtrain = xgb.DMatrix(as.matrix(x_train), label=y_train)
dtest = xgb.DMatrix(as.matrix(x_test))


# set the parameters of xgb
xgb_params = list(
        seed = 0,
        colsample_bytree = 0.5,
        subsample = 0.8,
        eta = 0.05, # replace this with 0.01 for local run to achieve 1113.93
        objective = 'reg:linear',
        max_depth = 12,
        alpha = 1,
        gamma = 2,
        min_child_weight = 1,
        base_score = 7.76
)


xg_eval_mae <- function (yhat, dtrain) {
        y = getinfo(dtrain, "label")
        err= mae(exp(y),exp(yhat) )
        return (list(metric = "error", value = err))
}




best_nrounds = 545 # comment this out when doing local 1113 run

# Train data
gbdt = xgb.train(xgb_params, dtrain, nrounds=as.integer(best_nrounds/0.8))

# Write data

submission = fread(SUBMISSION_FILE, colClasses = c("integer", "numeric"))
submission$loss = exp(predict(gbdt,dtest)) - SHIFT
write.csv(submission,'xgb_starter_v7.sub.csv',row.names = FALSE)
