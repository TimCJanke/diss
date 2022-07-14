import numpy as np
from mvpreg.mvpreg.data import data_utils
from mvpreg.mvpreg.evaluation import scoring_rules
from mvpreg.mvpreg import ScoringRuleDGR as DGR
from mvpreg.mvpreg import AdversarialDGR as GAN



data = data_utils.fetch_wind_spatial()
dates = data["dates"]
x = np.reshape(data["X"], (data["X"].shape[0], -1))
y = data["y"]
x_val = x[4200:5000,:]
y_val = y[4200:5000,:]
x_train = x[0:4200,:]
y_train = y[0:4200,:]

# data = data_utils.fetch_load()
# dates = data["dates"]
# x = np.reshape(data["X"], (data["X"].shape[0], -1))
# y = data["y"]
# # we only need the dummy varaibles once:
# feature_names = data["features"]
# feature_list = feature_names*data["X"].shape[1]

# drops_DoW =  [i for i,s in enumerate(feature_list) if "DoW" in s]
# drops_DoW = drops_DoW[int(len(drops_DoW)/data["X"].shape[1]):]

# drops_MoY = [i for i,s in enumerate(feature_list) if "MoY" in s]
# drops_MoY = drops_MoY[int(len(drops_MoY)/data["X"].shape[1]):]

# drops=[*drops_DoW, *drops_MoY]
# x = np.delete(x, drops, axis=1)

# x_val = x[1000:1500,:]
# y_val = y[1000:1500,:]
# x_train = x[0:1000,:]
# y_train = y[0:1000,:]


nn_base_config = {"n_layers": 3,
                "n_neurons": 200,
                "activation": "relu",
                "output_activation": "linear",
                "input_scaler": "Standard",
                "censored_left": 0.0,
                "censored_right": 1.0
                }

y_pred = {}
#%%
mdl_es=DGR(dim_in=x.shape[1], 
        dim_out=y.shape[1], 
         n_samples_val=100,
         n_samples_train=10,
         dim_latent=20, 
         output_scaler="Standard", 
         conditioning="FiLM", 
         show_model_summary=True,
         loss="ES",
         **nn_base_config
         )

mdl_es.fit(x_train, y_train, epochs=200, x_val=x_val, y_val=y_val, early_stopping=True, verbose=1, patience=10)
y_pred["ES"] = mdl_es.simulate(x_val, 100)

#%%
mdl_vs=DGR(dim_in=x.shape[1], 
        dim_out=y.shape[1], 
         n_samples_val=100,
         n_samples_train=10,
         dim_latent=20, 
         output_scaler="Standard", 
         conditioning="FiLM", 
         show_model_summary=True,
         loss="VS",
         p_vs=1.0,
         **nn_base_config
         )

mdl_vs.fit(x_train, y_train, epochs=200, x_val=x_val, y_val=y_val, early_stopping=True, verbose=1, patience=10)
y_pred["VS"] = mdl_vs.simulate(x_val, 100)

#%%
mdl_gan=GAN(dim_in=x.shape[1], 
        dim_out=y.shape[1], 
         n_samples_val=100,
         dim_latent=20, 
         output_scaler="Standard", 
         conditioning="FiLM", 
         show_model_summary=True,
         optimizer_kwargs={"beta_1": 0.5, "learning_rate": 0.0001},
         optimizer_discriminator_kwargs= {"beta_1": 0.5, "learning_rate": 0.0001},
         **nn_base_config
         )

mdl_gan.fit(x_train, y_train, epochs=500, x_val=x_val, y_val=y_val, early_stopping=True, verbose=1, patience=10)
y_pred["GAN"] = mdl_gan.simulate(x_val, 100)


#%%
ES={}
VS={}
for k in y_pred:
    ES[k] = scoring_rules.es_sample(y_val, y_pred[k])
    VS[k] = scoring_rules.vs_sample(y_val, y_pred[k])

#%%
import matplotlib.pyplot as plt

i=200
for k in y_pred:
    plt.figure()
    for s in range(50):
        plt.plot(y_pred[k][i,:,s], linewidth=1.0, alpha=0.6)
    plt.plot(y_val[i,:], color="black", linewidth=3)
    plt.title(k)
    
i=500
plt.figure()
for i in range(50):
    plt.plot(np.mean(y_pred["VS"][i,:,:], axis=-1), linewidth=0.5, color="red")
    plt.plot(y_val[i,:], color="black", linewidth=0.5)
