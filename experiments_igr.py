#%%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
from mvpreg.mvpreg import DeepQuantileRegression as QR
from mvpreg.mvpreg import DeepParametricRegression as PRM
from mvpreg.mvpreg import ScoringRuleDGR as DGR
from mvpreg.mvpreg import AdversarialDGR as GAN

from helpers import run_experiment, analyze_experiment

# data set
data_set_config = {"name": "wind_spatial",
                    "fetch_data":{"zones": list(np.arange(1,11)),
                                    "features": ['WE10', 'WE100', 'WD10', 'WD100', 'WD_difference', 'WS_ratio'],
                                    "hours": list(np.arange(0,24))},
                    "datetime_idx": None, #pd.date_range(start="2012/2/1 00:00", end="2012/5/1 23:00", freq='H'),               
                    "n_total": None,
                    "n_train": 24*7*25,
                    "n_val": 24*7*1,
                    "n_test": 24*7*1,
                    "n_samples_predict": 1000,
                    "early_stopping": True,
                    "epochs": 1000,
                    }

# shared configs for core NN
nn_base_config = {"n_layers": 3,
                "n_neurons": 200,
                "activation": "relu",
                "output_activation": None,
                "censored_left": 0.0, 
                "censored_right": 1.0, 
                "input_scaler": "Standard",
                #"output_scaler": None
                }

# configs for each model
model_configs={}
model_configs["LogitN"] = {"class": PRM,
                        "config_fixed": {**nn_base_config, 
                                            "distribution": "LogitNormal",
                                            "output_scaler": None
                                            },
                        "config_var": {}
                        }

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
                                        "n_samples_val": 200,
                                        "dim_latent": 20,
                                        "output_scaler": "Standard"
                                        },
                        "config_var": {"conditioning": ["concatenate", "FiLM"]
                                        }
                        }

model_configs["GAN"] = {"class": GAN,
                        "config_fixed": {**nn_base_config, 
                                        "n_samples_val": 200,
                                        "dim_latent": 20,
                                        "output_scaler": "Standard",
                                        "label_smoothing": 0.1,
                                        "optimizer_kwargs": {"beta_1": 0.5, "learning_rate": 0.0001},
                                        "optimizer_discriminator_kwargs": {"beta_1": 0.5, "learning_rate": 0.0001},
                                        },
                        
                        "config_var": {"conditioning": ["concatenate", "FiLM"]}
                        }

#%%
path_to_results = run_experiment(data_set_config, model_configs, copulas=["independence", "gaussian", "r-vine"],  name="25-1-1_rolling_window")


#%%
analyze_experiment(path_to_results)
