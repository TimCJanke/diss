#%%
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import pickle
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from mvpreg.mvpreg.data import data_utils
from mvpreg.mvpreg import DeepQuantileRegression as DQR
from mvpreg.mvpreg import DeepParametricRegression as PRM
from mvpreg.mvpreg import ScoringRuleDGR as DGR
from mvpreg.mvpreg import AdversarialDGR as CGAN

from mvpreg.mvpreg.evaluation import scoring_rules
from mvpreg.mvpreg.evaluation import visualization

#tf.get_logger().setLevel('WARNING')
#tf.autograph.set_verbosity(1)


#%% Data and hyperparams

config_wind_spatial = {"data":{
                            "zones": [1,2],#np.arange(1,11),
                            "features": ['WE10', 'WE100', 'WD10', 'WD100', 'WD_difference', 'WS_ratio']},
                       
                       "train_test":{"n_train": 24*25*7,
                                         "n_val": 24*2*7,
                                         "n_test": 24*7},
                        }

config = config_wind_spatial
data_set = "wind_spatial"

path = "results/"+str(data_set)+"/"+datetime.now().strftime("%Y%m%d%H%M")
os.makedirs(path)
#%% get data
#get data set
if data_set == "wind_spatial":
    data = data_utils.fetch_wind_spatial(**config["data"])
    #features = data["features"]
    x = np.reshape(data["X"], (data["X"].shape[0], -1))
    y = data["y"]


    # misc
    nn_base_config = {"dim_in": x.shape[1],
                    "dim_out": y.shape[1],
                    "n_layers": 3,
                    "n_neurons": 200,
                    "activation": "relu",
                    "output_activation": None,
                    "censored_left": 0.0, 
                    "censored_right": 1.0, 
                    "input_scaler": "Standard",
                    "output_scaler": None}

    param_config_fixed = {} 
    param_config_var = {"distribution": ["Normal", "LogitNormal"]}

qr_config_fixed = {"taus": np.arange(0.025,1.0, 0.025)}
copulas = ["independence", "gaussian", "r-vine"]


# DGR
dgr_config_fixed = {"n_samples_train": 10,
              "n_samples_val": 200,
              "dim_latent": y.shape[1]*2}
dgr_config_var={"conditioning": ["concatenate", "FiLM"], "loss": ["ES", "VS"]}


#GAN
gan_config_fixed = {**dgr_config_fixed,
                     "optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
                     "optimizer_discriminator": tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
                     "label_smoothing": 0.1}
_=gan_config_fixed.pop("n_samples_train")
gan_config_var={"conditioning": ["concatenate", "FiLM"]}




#%%
nb_train = config["train_test"]["n_train"]+config["train_test"]["n_val"]
nb_test = config["train_test"]["n_test"]

# ts_splitter = TimeSeriesSplit(n_splits=int(np.floor((len(x)-nb_train)/nb_test)),
#                               max_train_size=config["train_test"]["n_train"]+config["train_test"]["n_val"],
#                               test_size=config["train_test"]["n_test"])

ts_splitter = TimeSeriesSplit(n_splits=2,
                              max_train_size=config["train_test"]["n_train"]+config["train_test"]["n_val"],
                              test_size=config["train_test"]["n_test"])


y_predict = {}
y_test = []

pbar = tqdm(total=ts_splitter.get_n_splits())
for i, (idx_train_val, idx_test) in enumerate(ts_splitter.split(x)):
        
    idx_train = idx_train_val[0:config["train_test"]["n_train"]]
    idx_val = idx_train_val[config["train_test"]["n_train"]:]

    fit_dict={"x": x[idx_train],
              "y": y[idx_train],
              "x_val": x[idx_val],
              "y_val": y[idx_val], 
              "epochs": 1000,
              "early_stopping": True,
              "patience": 20,
              "plot_learning_curve": False}


    predict_dict={"x": x[idx_test],
                  "n_samples": 1000}

    ### test data ###
    y_test.append(y[idx_test])
    y_predict["DATA"].append(np.repeat(np.transpose(np.expand_dims(y[idx_train][np.random.choice(np.arange(0,len(idx_train)), np.minimum(predict_dict["n_samples"],len(y[idx_train])), replace=False),...],axis=0), (0,2,1)),repeats=len(idx_test),axis=0))

    
    ### QR ####
    model_qr = DQR(**nn_base_config, **qr_config_fixed)
    model_qr.fit(**fit_dict, fit_copula_model=False)
    
    for c in copulas:
        model_qr.copula_type = c
        model_qr.fit_copula(fit_dict["x"], fit_dict["y"])
        if i==0:
            y_predict["QR_"+c] = [model_qr.simulate(**predict_dict)]
        else:
            y_predict["QR_"+c].append(model_qr.simulate(**predict_dict))
        
    
    ### parametric ###
    for config_tmp in ParameterGrid(dgr_config_var):
        model_param = PRM(**nn_base_config, **param_config_fixed, **config_tmp)
        if model_param.distribution == "LogitNormal":
            fit_dict_lognorm = {**fit_dict}
            fit_dict_lognorm["y"] = np.clip(fit_dict["y"], 0.0+1e-3, 1.0-1e-3)
            fit_dict_lognorm["y_val"] = np.clip(fit_dict["y_val"], 0.0+1e-3, 1.0-1e-3)
            model_param.fit(**fit_dict_lognorm, fit_copula_model=False)
        else:
            model_param.fit(**fit_dict, fit_copula_model=False)
    
            for c in copulas:
                model_param.copula_type = c
                model_param.fit_copula(fit_dict["x"], np.clip(fit_dict["y"], 0.0+1e-3, 1.0-1e-3))
                nme_tmp="PARAM_"+'_'.join(map(str, list(config_tmp.values())))+"_"+c
                if i==0:
                    y_predict[nme_tmp] = [model_param.simulate(**predict_dict)]
                else:
                    y_predict[nme_tmp].append(model_param.simulate(**predict_dict))


    ### ScoringRule ####
    for config_tmp in ParameterGrid(dgr_config_var):
        model_dgr = DGR(**nn_base_config, **dgr_config_fixed, **config_tmp)
        model_dgr.fit(**fit_dict)
        if i==0:
            y_predict["DGR_"+'_'.join(map(str, list(config_tmp.values())))]=model_dgr.simulate(**predict_dict)
        else:
            y_predict["DGR_"+'_'.join(map(str, list(config_tmp.values())))].append(model_dgr.simulate(**predict_dict))


    ### GAN ####
    name = "CGAN"
    config_fixed=
    model = globals()[name]
    mdls=[]
    for config_tmp in ParameterGrid(gan_config_var):
        model =  CGAN(**nn_base_config, **gan_config_fixed, **config_tmp)
        mdls.append(model)
        model.fit(**fit_dict)
        mdl_name = name+"_"+'_'.join(map(str, list(config_tmp.values())))
        if i==0:
            y_predict[mdl_name]=model.simulate(**predict_dict)
        else:
            y_predict[mdl_name].append(model.simulate(**predict_dict))

    pbar.update()
pbar.close()



    
y_test = np.concatenate(y_test, axis=0)
pd.DataFrame(y_test).to_csv(path+"/y_test.csv", index=False)

for key in y_predict:
    y_predict[key] = np.concatenate(y_predict[key], axis=0)
    
### save results ###
with open(path+"/results.pkl", 'wb') as f:
    pickle.dump(y_predict, f)

print("\ntraining completed.")

#%% evalutaion
print("\ncomputing scores...")
# let's compare models based on several scores
scores={}
for key in y_predict:
    scores[key] = scoring_rules.all_scores_mv_sample(y_test, y_predict[key], 
                                                      return_single_scores=True,
                                                      CALIBRATION =True,
                                                      MSE=True,
                                                      MAE=True, 
                                                      CRPS=True, 
                                                      ES=True, 
                                                      VS05=True, 
                                                      VS1=True, 
                                                      CES=False, 
                                                      CVS05=False, 
                                                      CVS1=False)
### save results ###
with open(path+"/score_series.pkl", 'wb') as f:
    pickle.dump(scores, f)
    
mean_scores=pd.DataFrame()
for mdl_key in scores:
    for score_key in scores[mdl_key]:
        mean_scores.loc[mdl_key, score_key] = np.mean(scores[mdl_key][score_key])
print(mean_scores)

### save results ###
with open(path+"/scores_mean.pkl", 'wb') as f:
    pickle.dump(mean_scores, f)

print("\nAll done.")


# # let's assess significance of score differences via DM test and plot the results
# dm_test_results={}
# score_names = list(scores[list(y_predict.keys())[0]].keys())
# for score_key in score_names:
#     if score_key != "CAL":
#         score_series={}
#         for mdl_key in scores:
#             score_series[mdl_key] = scores[mdl_key][score_key]
#         dm_test_results[score_key] = scoring_rules.dm_test_matrix(score_series)
#         #visualization.plot_dm_test_matrix(dm_test_results[score_key], title=score_key)

# ### save results ###
# with open(path+"/dm_tests.pkl", 'wb') as f:
#     pickle.dump(dm_test_results, f)



from sklearn.model_selection import ParameterGrid
dgr_config_var={"conditioning": ["concatenate", "FiLM"], "loss": ["ES", "VS"], "n_samples": [1]}
list(ParameterGrid(dgr_config_var))

for i in ParameterGrid(param_config_var):
    print(i)
    #print(list(i.values()))