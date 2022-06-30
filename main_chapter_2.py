#%%
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import pickle
import os
from datetime import datetime

from mvpreg.mvpreg.data import data_utils
from mvpreg.mvpreg import DeepQuantileRegression as DQR
from mvpreg.mvpreg import DeepParametricRegression as PRM
from mvpreg.mvpreg import ScoringRuleDGR as DGR
from mvpreg.mvpreg import AdversarialDGR as CGAN

from mvpreg.mvpreg.evaluation import scoring_rules
from mvpreg.mvpreg.evaluation import visualization


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


    
    param_config ={"distribution": "LogitNormal",
                  "copula_type": "gaussian"}
    
qr_config ={"taus": np.arange(0.05,1.0, 0.05),
            "copula_type": "gaussian"}

dgr_config = {"n_samples_train": 10,
              "n_samples_val": 200,
              "dim_latent": y.shape[1],
              "conditioning": "FiLM"}

gan_config = {**dgr_config,
             "optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
             "optimizer_discriminator": tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
             "label_smoothing": 0.1}
_=gan_config.pop("n_samples_train")

nb_train = config["train_test"]["n_train"]+config["train_test"]["n_val"]
nb_test = config["train_test"]["n_test"]

# ts_splitter = TimeSeriesSplit(n_splits=int(np.floor((len(x)-nb_train)/nb_test)),
#                               max_train_size=config["train_test"]["n_train"]+config["train_test"]["n_val"],
#                               test_size=config["train_test"]["n_test"])

ts_splitter = TimeSeriesSplit(n_splits=2,
                              max_train_size=config["train_test"]["n_train"]+config["train_test"]["n_val"],
                              test_size=config["train_test"]["n_test"])

#%%
y_predict = {"DATA":[],
             "QR":[],
             "PARAM":[],
             "DGR":[],
             "GAN":[]}
y_test = []
for i, (idx_train_val, idx_test) in enumerate(ts_splitter.split(x)):
    idx_train = idx_train_val[0:config["train_test"]["n_train"]]
    idx_val = idx_train_val[config["train_test"]["n_train"]:]


    fit_dict={"x": x[idx_train],
              "y": y[idx_train],
              "x_val": x[idx_val],
              "y_val": y[idx_val], 
              "epochs":1000,
              "early_stopping":True,
              "patience": 20,
              "plot_learning_curve": False}


    predict_dict={"x": x[idx_test],
                  "n_samples": 1000}

    ### test data ###
    y_test.append(y[idx_test])
    y_predict["DATA"].append(np.repeat(np.transpose(np.expand_dims(y[idx_train][np.random.choice(np.arange(0,len(idx_train)), np.minimum(predict_dict["n_samples"],len(y[idx_train])), replace=False),...],axis=0), (0,2,1)),repeats=len(idx_test),axis=0))

    
    ### QR ####
    model_qr = DQR(**nn_base_config, **qr_config)
    model_qr.fit(**fit_dict)
    y_predict["QR"].append(model_qr.simulate(**predict_dict))
    
    
    ### parametric ###
    model_param = PRM(**nn_base_config, **param_config)    
    if param_config["distribution"] == "LogitNormal":
        model_param.fit(x=fit_dict["x"], 
                        y=np.clip(fit_dict["y"], 0.0+1e-3, 1.0-1e-3), 
                        x_val=fit_dict["x_val"],
                        y_val=np.clip(fit_dict["y_val"], 0.0+1e-3, 1.0-1e-3),
                        epochs=fit_dict["epochs"],
                        early_stopping=fit_dict["early_stopping"],
                        patience=fit_dict["patience"],
                        plot_learning_curve=fit_dict["plot_learning_curve"])
    else:
        model_param.fit(**fit_dict)
    y_predict["PARAM"].append(model_param.simulate(**predict_dict))


    ### ScoringRule ####
    model_dgr = DGR(**nn_base_config, **dgr_config)
    model_dgr.fit(**fit_dict)
    y_predict["DGR"].append(model_dgr.simulate(**predict_dict))


    ### CGAN ###
    model_gan = CGAN(**nn_base_config, **gan_config)
    model_gan.fit(**fit_dict)
    y_predict["GAN"].append(model_gan.simulate(**predict_dict))
    

y_test = np.concatenate(y_test)
pd.DataFrame(y_test).to_csv(path+"/y_test.csv", index=False)

for key in y_predict:
    y_predict[key] = np.concatenate(y_predict[key], axis=0)
    
### save results ###
with open(path+"/results.pkl", 'wb') as f:
    pickle.dump(y_predict, f)


#%% evalutaion

# let's compare models based on several scores
# scores={}
# for key in y_predict:
#     scores[key] = scoring_rules.get_all_scores_sample(y_test, y_predict[key], return_single_scores=True)
    
    
# mean_scores = pd.DataFrame(scores).T
# print(scores)

# let's assess significance of score differences via DM test and plot the results
# es_series={}
# for key in y_predict:
#     es_series[key] = scoring_rules.es_sample(y_test, y_predict[key][:,:,0:500], return_single_scores=True)

# dm_results_matrix = scoring_rules.dm_test_matrix(es_series)
# visualization.plot_dm_test_matrix(dm_results_matrix)