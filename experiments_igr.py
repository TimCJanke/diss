# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:32:38 2022

@author: Tim_Janke
"""

#%%
import numpy as np
import pathlib
import json
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
from mvpreg.mvpreg import DeepQuantileRegression as QR
from mvpreg.mvpreg import DeepParametricRegression as PRM
from mvpreg.mvpreg import ScoringRuleDGR as DGR
from mvpreg.mvpreg import AdversarialDGR as GAN

from mvpreg.mvpreg.evaluation import scoring_rules
#from mvpreg.mvpreg.evaluation import visualization

#%%
def run_experiment(data_set_config, 
                   model_configs, 
                   copulas = ["independence", "gaussian", "r-vine"], 
                   path=None, 
                   name=None, 
                   losses={"taus": [0.05, 0.2, 0.8, 0.95],
                            "CALIBRATION": True,
                            "MSE": True,
                            "MAE": True,
                            "PB": True,
                            "CRPS": True, 
                            "ES": True, 
                            "VS05": True, 
                            "VS1": True, 
                            "CES": False, 
                            "CVS05": False, 
                            "CVS1": False},
                   validation_loss="ES"):
    
    ### dir to save the results ###
    if path is None:
        path = "results/"+str(data_set_config["name"])+"/"
        if name is None:
            path = path+datetime.now().strftime("%Y%m%d%H%M")
        else:
            path = path+name
    if os.path.exists(path):
        path = path+datetime.now().strftime("%Y%m%d%H%M") 
    os.makedirs(path)


    #### fetch data set ###
    data_set_name = data_set_config["name"]
    if data_set_name == "wind_spatial":
        data = data_utils.fetch_wind_spatial(**data_set_config["fetch_data"])
        dates = data["dates"]
        x = np.reshape(data["X"], (data["X"].shape[0], -1))
        y = data["y"]
    else:
        raise ValueError(f"Unknown data set: {data_set_name}")
    
    
    ### prepare data set ###
    if data_set_config["datetime_idx"] is not None:
        x = pd.DataFrame(x, index=dates)
        x = x.loc[data_set_config["datetime_idx"]]
        
        y = pd.DataFrame(y, index=dates)
        y = y.loc[data_set_config["datetime_idx"]]
        
        dates = y.index
        x = x.values
        y = y.values
    
    N = data_set_config["n_total"]
    if N is not None:
        x = x[0:N, ...]
        y = y[0:N, ...]
        dates = dates[0:N]
    
    n_train = data_set_config["n_train"]
    n_train_val = data_set_config["n_train"]+data_set_config["n_val"]
    n_test = data_set_config["n_test"]
    
    ts_splitter = TimeSeriesSplit(int(np.floor((len(y)-n_train_val)/n_test)),
                                  max_train_size=n_train_val,
                                  test_size=n_test)
    
    
    ### iterate over data set splits ###
    y_predict = {key: {} for key in model_configs.keys()}
    pbar = tqdm(total=ts_splitter.get_n_splits())
    for i, (idx_train_val, idx_test) in enumerate(ts_splitter.split(x)):
        
        # prepare inputs
        idx_train = idx_train_val[0:n_train]
        idx_val = idx_train_val[n_train:]    
        fit_dict={"x": x[idx_train],
                  "y": y[idx_train],
                  "x_val": x[idx_val],
                  "y_val": y[idx_val], 
                  "epochs": data_set_config["epochs"],
                  "early_stopping": data_set_config["early_stopping"],
                  "patience": 20,
                  "plot_learning_curve": False}    
        predict_dict={"x": x[idx_test],"n_samples": data_set_config["n_samples_predict"]}
        
        
        ### groundtruth data and baseline ###
        if i==0:
            dates_test = dates[idx_test]
            y_test = [y[idx_test]]
            y_predict["DATA"] = {}
            y_predict["DATA"]["train"] = [np.repeat(np.transpose(np.expand_dims(y[idx_train][np.random.choice(np.arange(0,len(idx_train)), np.minimum(data_set_config["n_samples_predict"],len(y[idx_train])), replace=False),...],axis=0), (0,2,1)),repeats=len(idx_test),axis=0)]
        else:
            dates_test = dates_test.union(dates[idx_test])
            y_test.append(y[idx_test])
            y_predict["DATA"]["train"].append(np.repeat(np.transpose(np.expand_dims(y[idx_train][np.random.choice(np.arange(0,len(idx_train)), np.minimum(data_set_config["n_samples_predict"],len(y[idx_train])), replace=False),...],axis=0), (0,2,1)),repeats=len(idx_test),axis=0))

        
        ### iterate over models ###
        config_combinations={}
        for mdl_name, mdl_cnfg in model_configs.items():
            config_combinations[mdl_name] = {}
            model_class = mdl_cnfg["class"]
            
            # iterate over all parameter config combinations for this model
            for config_tmp in ParameterGrid(mdl_cnfg["config_var"]):
                
                mdl_id = "_".join(["("+k+":"+v+")" for k,v in list(zip(list(map(str, list(config_tmp.keys()))), list(map(str, list(config_tmp.values())))))])
                model_tmp =  model_class(dim_in=x.shape[1], dim_out=y.shape[1], **mdl_cnfg["config_fixed"], **config_tmp) # init model
                
                # handle special case of LogitNormal distribution on (0,1)
                if hasattr(model_tmp, "distribution"):
                    if model_tmp.distribution== "LogitNormal":
                        fit_dict_ = {**fit_dict}
                        fit_dict_["y"] = np.clip(fit_dict_["y"], 0.0+1e-3, 1.0-1e-3)
                        fit_dict_["y_val"] = np.clip(fit_dict_["y_val"], 0.0+1e-3, 1.0-1e-3)
                    else:
                        fit_dict_ = fit_dict
                else:
                    fit_dict_ = fit_dict
                
                # fit model
                model_tmp.fit(**fit_dict_)
                
                # simulate for different copulas if model has one
                if hasattr(model_tmp, "copula"):
                    for c in copulas:
                        model_tmp.copula_type = c
                        model_tmp.fit_copula(fit_dict_["x"], fit_dict_["y"])
                        
                        if i==0:
                            y_predict[mdl_name][mdl_id+"_"+"(copula:"+c+")"]=[model_tmp.simulate(**predict_dict)]
                        else:
                            y_predict[mdl_name][mdl_id+"_"+"(copula:"+c+")"].append(model_tmp.simulate(**predict_dict))
                        
                        config_combinations[mdl_name][mdl_id+"_"+"(copula:"+c+")"] = {**config_tmp, "copula": c}
                        
                else:
                    if i==0:
                        y_predict[mdl_name][mdl_id]=[model_tmp.simulate(**predict_dict)]
                    else:
                        y_predict[mdl_name][mdl_id].append(model_tmp.simulate(**predict_dict))
                    
                    config_combinations[mdl_name][mdl_id] = config_tmp
         
        # save intermediate results
        # pd.DataFrame(np.concatenate(y_test, axis=0), index=dates_test).to_csv(path+"/y_test.csv")
        # with open(path+"/results.pkl", 'wb') as f:
        #     pickle.dump({key: np.concatenate(value) for key, value in y_predict.items()}, f)
        
        pbar.update()
    
    pbar.close()        
        
    
    y_test = np.concatenate(y_test, axis=0)
    pd.DataFrame(y_test, index=dates_test).to_csv(path+"/y_test.csv")
    
    # concatenate the lists to arrays
    for mdl_nm,v in y_predict.items():
        for mdl_cnfg in v:
            y_predict[mdl_nm][mdl_cnfg] = np.concatenate(y_predict[mdl_nm][mdl_cnfg], axis=0)
    
    ### save results ###
    with open(path+"/results.pkl", 'wb') as f:
        pickle.dump(y_predict, f)
        
    # with open(path+"/model_configs.pkl", "wb") as f:
    #     pickle.dump(model_configs, f)

    with open(path+"/data_set_config.pkl", "wb") as f:
        pickle.dump(data_set_config, f)
    
    scores = {}
    for mdl_nm,v in y_predict.items():
        scores[mdl_nm] = {}
        for mdl_cnfg in v:
            scores[mdl_nm][mdl_cnfg] = scoring_rules.all_scores_mv_sample(y_test, y_predict[mdl_nm][mdl_cnfg], 
                                                                          return_single_scores=True,
                                                                          **losses)
            
    # per model type get best performing hyperparameters
    optimal_configs={}
    for mdl_nm in config_combinations:
        loss={}
        for mdl_cnfg in config_combinations[mdl_nm]:
            loss[mdl_cnfg] = np.mean(scores[mdl_nm][mdl_cnfg][validation_loss])
        
        best_cnfg_id = min(loss, key=loss.get) # key smallest ES loss
        optimal_configs[mdl_nm] = config_combinations[mdl_nm][best_cnfg_id]
    
    print(optimal_configs)
    
    with open(path+"/optimal_configs.pkl", 'wb') as f:
        pickle.dump(optimal_configs, f)
            
    scores_flat = {}
    for mdl_nm in scores:
        for mdl_id in scores[mdl_nm]:
            scores_flat[mdl_nm+"---"+mdl_id] = scores[mdl_nm][mdl_id]
    scores = scores_flat

    y_predict_flat = {}
    for mdl_nm in y_predict:
        for mdl_id in y_predict[mdl_nm]:
            y_predict_flat[mdl_nm+"---"+mdl_id] = y_predict[mdl_nm][mdl_id]
    y_predict = y_predict_flat

    
    ### save results ###
    with open(path+"/score_series.pkl", 'wb') as f:
        pickle.dump(scores, f)

    
    mean_scores=pd.DataFrame()
    for mdl_key in scores:
        for score_key in scores[mdl_key]:
            mean_scores.loc[mdl_key, score_key] = np.mean(scores[mdl_key][score_key])
    mean_scores.to_csv(path+"/mean_scores.csv")
    
    # assess significance of score differences via DM test
    dm_test_results={}
    score_names = list(scores[list(y_predict.keys())[0]].keys())
    for score_key in score_names:
        if score_key != "CAL":
            score_series={}
            for mdl_key in scores:
                score_series[mdl_key] = scores[mdl_key][score_key]
            dm_test_results[score_key] = scoring_rules.dm_test_matrix(score_series)
            #visualization.plot_dm_test_matrix(dm_test_results[score_key], title=score_key)
    
    ### save results ###
    with open(path+"/dm_test_results.pkl", 'wb') as f:
        pickle.dump(dm_test_results, f)
    
    print("\nexperiment completed.")


#%% get data
if __name__ == "__main__":

    # data set
    data_set_config = {"name": "wind_spatial",
                       "fetch_data":{"zones": [4,5,6,1,7,8],#list(np.arange(1,11)),
                                     "features": ['WE10', 'WE100', 'WD10', 'WD100', 'WD_difference', 'WS_ratio'],
                                     "hours": list(np.arange(0,24))},
                       "datetime_idx": None, #pd.date_range(start="2012/2/1 00:00", end="2012/5/1 23:00", freq='H'),               
                       "n_total": 24*24*7,
                       "n_train": 24*12*7,
                       "n_val": 24*4*7,
                       "n_test": 24*4*7,
                       "n_samples_predict": 250,
                       "early_stopping": True,
                       "epochs": 50,
                       }

    # data_set_config = {"name": "wind_spatial",
    #                    "fetch_data":{"zones": [1, 2, 3],#np.arange(1,11),
    #                                  "features": ['WE10', 'WE100', 'WD10', 'WD100', 'WD_difference', 'WS_ratio'],
    #                                  "hours": list(np.arange(0,24))},
    #                    "datetime_idx": pd.date_range(start="2012/2/1 00:00", end="2012/5/1 23:00", freq='H'),
    #                    "n_total": None,
    #                    "n_train": 24*10*7,
    #                    "n_val": 24*7,
    #                    "n_test": 24*7,
    #                    "n_samples_predict": 10,
    #                    "early_stopping": True,
    #                    "epochs": 5,
    #                    }
    
    # shared configs for core NN
    nn_base_config = {"n_layers": 3,
                    "n_neurons": 200,
                    "activation": "relu",
                    #"output_activation": None,
                    "censored_left": 0.0, 
                    "censored_right": 1.0, 
                    #"input_scaler": "Standard",
                    #"output_scaler": None
                    }
    
    # configs for each model
    model_configs={}
    model_configs["LogitN"] = {"class": PRM,
                            "config_fixed": {**nn_base_config, 
                                             "distribution":"LogitNormal",
                                             },
                            "config_var": {"input_scaler": ["Standard", "MinMax"],
                                           }
                            }

    model_configs["Normal"] = {"class": PRM,
                            "config_fixed": {**nn_base_config, 
                                             "distribution":"Normal"
                                             },
                            "config_var": {"input_scaler": ["Standard", "MinMax"],
                                           "output_scaler": [None, "Standard"]
                                           }
                            }
        
    model_configs["QR"] = {"class": QR,
                            "config_fixed": {**nn_base_config, "taus": list(np.round(np.arange(0.025,1.0, 0.025), 4))},
                            "config_var": {"input_scaler": ["Standard", "MinMax"],
                                           "output_scaler": [None, "Standard"],
                                           "output_activation": ["linear", "sigmoid"]
                                           }
                            }
    
    # model_configs["DGR"] = {"class": DGR,
    #                         "config_fixed": {**nn_base_config, 
    #                                         "n_samples_train": 10,
    #                                         "n_samples_val": 200,
    #                                         "dim_latent": 20
    #                                         },
    #                         "config_var": {"conditioning": ["concatenate", "FiLM"],
    #                                        "input_scaler": ["Standard", "MinMax"],
    #                                        "output_scaler": [None, "Standard"],
    #                                        "output_activation": ["linear", "sigmoid"]
    #                                        }
    #                         }
    
    # model_configs["GAN"] = {"class": GAN,
    #                         "config_fixed": {**nn_base_config, 
    #                                         "n_samples_val": 200,
    #                                         "dim_latent": 20,
    #                                         #"optimizer": tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
    #                                         "optimizer": "Adam",
    #                                         #"optimizer_discriminator": tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.1),
    #                                         "label_smoothing": 0.1
    #                                         },
                            
    #                         "config_var": {"conditioning": ["concatenate", "FiLM"],
    #                                         "optimizer_kwargs": [{"beta_1": 0.0}, {"learning_rate": 0.0001}],
    #                                         "optimizer_discriminator_kwargs": [{"beta_1": 0.0}, {"learning_rate": 0.0001}]}
    #                         }

    run_experiment(data_set_config, model_configs, copulas=["independence"],  name="cv_wind_spatial")
    

# d = {}
# y = {}
# for mdl_name, mdl_cnfg in model_configs.items():    
#     # iterate over all parameter config combinations for this model
#     d[mdl_name] = {}
#     y[mdl_name] = {}
#     for config_tmp in ParameterGrid(mdl_cnfg["config_var"]):
#         ks = list(map(str, list(config_tmp.keys())))
#         vls = list(map(str, list(config_tmp.values())))
        
#         mdl_id = mdl_name+"_"+'_'.join(map(str, list(config_tmp.values()))) # create model id
#         d[mdl_name][mdl_id] = config_tmp
#         y[mdl_name][mdl_id] = np.random.normal(size=(100,3,5))
        
# list(zip(list(map(str, list(config_tmp.keys()))), list(map(str, list(config_tmp.values())))))
# mdl_name+"---"+"_".join(["("+k+":"+v+")" for k,v in list(zip(list(map(str, list(config_tmp.keys()))), list(map(str, list(config_tmp.values())))))])