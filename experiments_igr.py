#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import argparse
import numpy as np
from mvpreg.mvpreg import DeepQuantileRegression as QR
from mvpreg.mvpreg import DeepParametricRegression as PRM
from mvpreg.mvpreg import ScoringRuleDGR as DGR
from mvpreg.mvpreg import AdversarialDGR as GAN
from helpers import run_experiment, analyze_experiment

parser = argparse.ArgumentParser()
parser.add_argument("data_set", type=str, choices=["wind_spatial", "solar_spatial", "load"])
parser.add_argument("--n_train", type=int, required=False)
parser.add_argument("--n_val", type=int, required=False)
parser.add_argument("--n_test", type=int, required=False)
parser.add_argument("--move_testset_window", type=int, required=False)
parser.add_argument("--tag", type=str, required=False)
args = parser.parse_args()

data_set = args.data_set

if args.n_train is not None:
    n_train = args.n_train
else:
    if data_set == "wind_spatial":
        n_train = 24*7*25
    elif data_set == "solar_spatial":
        n_train = 14*7*25
    elif data_set == "load":
        n_train = 365*4-28-7

if args.n_val is not None:
    n_val = args.n_val
else:
    if data_set == "wind_spatial":
        n_val = 24*7*1
    elif data_set == "solar_spatial":
        n_val = 14*7*1
    elif data_set == "load":
        n_val = 7*4

if args.n_test is not None:
    n_test = args.n_test
else:
    if data_set == "wind_spatial":
        n_test = 24*7*1
    elif data_set == "solar_spatial":
        n_test = 14*7*1
    elif data_set == "load":
        n_test = 7

if args.move_testset_window is not None:
    move_testset_window = args.move_testset_window
else:
    move_testset_window = True

if args.tag is not None:
    tag = args.tag
else:
    tag=None
    
    
print("\nRunning experiment with following options:")
print(f"data_set: {data_set}")
print(f"n_train: {n_train}")
print(f"n_val: {n_val}")
print(f"n_test: {n_test}")
print(f"move_testset_window: {move_testset_window}")
print(f"tag: {tag}\n")


if data_set == "wind_spatial":
# data set
    data_set_config = {"name": "wind_spatial",
                        "fetch_data":{"zones": list(np.arange(1,11)),
                                        "features": ['WE10', 'WE100', 'WD10', 'WD100', 'WD_difference', 'WS_ratio'],
                                        "hours": list(np.arange(0,24))},
                        "datetime_idx": None, #pd.date_range(start="2012/2/1 00:00", end="2012/5/1 23:00", freq='H'),               
                        "n_total": None,
                        "n_train": n_train,
                        "n_val": n_val,
                        "n_test": n_test,
                        "n_samples_predict": 1000,
                        "early_stopping": True,
                        "patience": 20,
                        "epochs": 500,
                        "moving_window": True
                        }
    dim_latent = len(data_set_config["fetch_data"]["zones"])
    
    # shared configs for core NN
    nn_base_config = {"n_layers": 3,
                    "n_neurons": 200,
                    "activation": "relu",
                    "output_activation": None,
                    "censored_left": 0.0, 
                    "censored_right": 1.0, 
                    "input_scaler": "Standard",
                    }

    # configs for each model
    model_configs={}
    
    # Linear paramertic model with Logit-Normal target dist
    model_configs["LogitNLinear"] = {"class": PRM,
                                    "config_fixed": {"n_layers": 0,
                                                    "output_activation": None,
                                                    "censored_left": 0.0, 
                                                    "censored_right": 1.0, 
                                                    "input_scaler": "Standard", 
                                                    "distribution": "LogitNormal",
                                                    "output_scaler": None,
                                                    },
                                    "config_var": {}
                                    }
    model_configs["LogitN"] = {"class": PRM,
                            "config_fixed": {**nn_base_config, 
                                                "distribution": "LogitNormal",
                                                "output_scaler": None
                                                },
                            "config_var": {}
                            }




if data_set == "solar_spatial":
    # data set
    data_set_config = {"name": "solar_spatial",
                        "fetch_data":{"zones": list(np.arange(1,4)),
                                        "features": ['T', 'SSRD', 'RH', 'WS10'],
                                        "hours": list(np.arange(6,20))},
                        "datetime_idx": None, #pd.date_range(start="2012/2/1 00:00", end="2012/5/1 23:00", freq='H'),               
                        "n_total": None,
                        "n_train": n_train,
                        "n_val": n_val,
                        "n_test": n_test,
                        "n_samples_predict": 1000,
                        "early_stopping": True,
                        "patience": 20,
                        "epochs": 500,
                        "moving_window": True
                        }
    dim_latent = len(data_set_config["fetch_data"]["zones"])

    # shared configs for core NN
    nn_base_config = {"n_layers": 3,
                    "n_neurons": 200,
                    "activation": "relu",#"elu",
                    "output_activation": None,
                    "censored_left": 0.0, 
                    "censored_right": 1.0, 
                    "input_scaler": "Standard",
                    }

    # configs for each model
    model_configs={}

    # Linear paramertic model with Logit-Normal target dist
    model_configs["LogitNLinear"] = {"class": PRM,
                                    "config_fixed": {"n_layers": 0,
                                                    "output_activation": None,
                                                    "censored_left": 0.0, 
                                                    "censored_right": 1.0,                                                    
                                                    "input_scaler": "Standard", 
                                                    "distribution": "LogitNormal",
                                                    "output_scaler": None,
                                                    },
                                    "config_var": {}
                                    }

    # Parametric model with LogitNormal targets
    model_configs["LogitN"] = {"class": PRM,
                            "config_fixed": {**nn_base_config, 
                                                "distribution": "LogitNormal",
                                                "output_scaler": None
                                                },
                            "config_var": {}
                            }


if data_set == "load":
    # data set
    data_set_config = {"name": "load",
                        "fetch_data":{"features": ['TEMP', 'DoW_DUMMY', 'MoY_DUMMY'],
                                      "load_lags": [1,2,7],
                                      "temp_lags": [1,2,7],
                                      "hours": list(np.arange(0,24))},
                        "datetime_idx": None, #pd.date_range(start="2012/2/1 00:00", end="2012/5/1 23:00", freq='H'),               
                        "n_total": None,
                        "n_train": n_train,
                        "n_val": n_val,
                        "n_test": n_test,
                        "n_samples_predict": 1000,
                        "early_stopping": True,
                        "patience": 20,
                        "epochs": 500,
                        "moving_window": True
                        }
    dim_latent = len(data_set_config["fetch_data"]["hours"])

    # shared configs for core NN
    nn_base_config = {"n_layers": 3,
                    "n_neurons": 200,
                    "activation": "relu",
                    "output_activation": None,
                    "input_scaler": "Standard",
                    }

    # configs for each model
    model_configs={}

    # Linear paramertic model with Normal target dist
    model_configs["NormalLinear"] = {"class": PRM,
                                    "config_fixed": {"n_layers": 0,
                                                    "output_activation": None,
                                                    "input_scaler": "Standard", 
                                                    "distribution": "Normal",
                                                    "output_scaler": "Standard",
                                                    },
                                    "config_var": {}
                                    }

    # Paramertic model with Normal target dist
    model_configs["Normal"] = {"class": PRM,
                            "config_fixed": {**nn_base_config, 
                                                "distribution": "Normal",
                                                "output_scaler": "Standard",
                                                },
                            "config_var": {}
                            }


model_configs["QR"] = {"class": QR,
                        "config_fixed": {**nn_base_config, 
                                        "taus": list(np.round(np.arange(0.025,1.0, 0.025), 4)),
                                        "output_scaler": "Standard"},
                        "config_var": {}
                        }


model_configs["DGR"] = {"class": DGR,
                        "config_fixed": {**nn_base_config, 
                                        "n_samples_train": 10,
                                        "n_samples_val": 100,
                                        "dim_latent": dim_latent,
                                        "output_scaler": "Standard",
                                        },
                        "config_var": {"conditioning": ["concatenate", "FiLM"],
                                        "loss": ["ES", "VS"]
                                        }
                        }

model_configs["GAN"] = {"class": GAN,
                        "config_fixed": {**nn_base_config, 
                                        "n_samples_val": 100,
                                        "dim_latent": dim_latent,
                                        "output_scaler": "Standard",
                                        "label_smoothing": 0.1,
                                        "optimizer_kwargs": {"beta_1": 0.0, "learning_rate": 0.0005},
                                        "optimizer_discriminator_kwargs": {"beta_1": 0.0, "learning_rate": 0.0005},
                                        },
                        
                        "config_var": {"conditioning": ["concatenate", "FiLM"]}
                        }



path_to_results = run_experiment(data_set_config, model_configs, copulas=["independence", "gaussian"], move_testset_window=move_testset_window, name=tag)

analyze_experiment(path_to_results)
