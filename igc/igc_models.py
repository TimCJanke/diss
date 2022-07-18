import numpy as np
from scipy.stats import rankdata
import tensorflow as tf
from tensorflow.keras import layers

from ..mvpreg.mvpreg.models.generative_regression import ScoringRuleDGR
from ..mvpreg.mvpreg.models.losses import EnergyScore, VariogramScore
from ..mvpreg.mvpreg.models.layers import FiLM, ClipValues, UnconditionalGaussianSampling


class ScoringRuleCopulaDGR(ScoringRuleDGR):
    def __init__(self,
                 train_with_softrank_layer=True,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.train_with_softrank_layer = train_with_softrank_layer
        
        self.model = self._build_model()
        if self.show_model_summary:
            self.model.summary(expand_nested=True)


    def _build_model(self):
        model = ScoringRuleCIGC(dim_out=self.dim_out, 
                                dim_latent = self.dim_latent,
                                n_layers_generator=self.n_layers, 
                                n_neurons_generator=self.n_neurons, 
                                activation_generator=self.activation,
                                output_activation_generator=self.output_activation,
                                conditioning_generator=self.conditioning,
                                FiLM_modulation=self.FiLM_modulation,
                                n_samples_train=self.n_samples_train,
                                n_samples_val=self.n_samples_val,
                                train_with_softrank_layer = self.train_with_softrank_layer)

        _ = model(np.random.normal(size=(1, self.dim_in, 1))) # need to do one forward pass to build model graph
        #model.summary(expand_nested=True) # now we can call summary
        model.compile(loss=self.loss, optimizer=self.optimizer)
        
        return model
    
    def simulate(self, x, n_samples=1, normalize=True, n_samples_normalize=None, softrank_normalize=False):
        """ draw n_samples randomly from conditional distribution p(y|x)"""
        
        if normalize is False:
            return self.model(np.repeat(np.expand_dims(self._scale_x(x), axis=2), repeats=n_samples, axis=2), training=False).numpy()
        
        if softrank_normalize:
            return self.model(np.repeat(np.expand_dims(self._scale_x(x), axis=2), repeats=n_samples, axis=2), training=True).numpy()            
        
        else:
            if n_samples_normalize is None:
                n_samples_normalize = np.maximum(200, n_samples*4)
            y = self.model(np.repeat(np.expand_dims(self._scale_x(x), axis=2), repeats=n_samples_normalize, axis=2), training=False).numpy()
            y = rankdata(y, axis=2)/(y.shape[2]+1)
            return y[:,:,np.random.choice(y.shape[2], n_samples)]


class ConditionalImplicitGenerativeCopula(tf.keras.Model):
    def __init__(self, 
                dim_out,
                n_layers_generator,
                n_neurons_generator,
                activation_generator,
                output_activation_generator="linear", 
                conditioning_generator="concatenate",
                FiLM_modulation="constant",
                dim_latent=None,
                n_samples_train=10, 
                n_samples_val=None,
                train_with_softrank_layer=True,
                **kwargs):
        
        super().__init__(**kwargs)

        # hyperparams
        self.n_samples_train = n_samples_train
        if n_samples_val is None:
            self.n_samples_val = n_samples_train
        else:
            self.n_samples_val = n_samples_val
        
        if dim_latent is None:
            dim_latent = dim_out

        self.generator_config = {"dim_out": dim_out,
                                 "dim_latent": dim_latent,
                                 "n_layers": n_layers_generator, 
                                 "n_neurons": n_neurons_generator, 
                                 "activation": activation_generator,
                                 "output_activation": output_activation_generator,
                                 "train_with_softrank_layer": train_with_softrank_layer}
        
        # init generator model
        if conditioning_generator == "concatenate":
            self.generator = ConcatGenerator(**self.generator_config)
            
        elif conditioning_generator == "FiLM":
            self.generator_config["modulation"] = FiLM_modulation
            self.generator = FiLMGenerator(**self.generator_config)

    
    def call(self, x, training):
        # 'x' is a tensor of shape (batch_size, dim_in, n_samples)
        # 'y' is a tensor of shape (batch_size, dim_out, n_samples)
        y = self.generator(x, training)
        return y


class ScoringRuleCIGC(ConditionalImplicitGenerativeCopula):
    def __init__(self,
                 dim_out,
                 n_layers_generator,
                 n_neurons_generator,
                 activation_generator,
                 n_samples_train=10, 
                 **kwargs):
        
        super().__init__(dim_out, 
                         n_layers_generator,
                         n_neurons_generator,
                         activation_generator,
                         **kwargs)

        # hyperparams
        self.n_samples_train = n_samples_train


    def train_step(self, data):
        x_train, y_train = data
        
        with tf.GradientTape() as tape:
            y_predict = self(tf.repeat(tf.expand_dims(x_train, axis=2), repeats=self.n_samples_train, axis=2), training=True)
            loss = self.compiled_loss(y_train, y_predict)
        grads = tape.gradient(loss, self.generator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.compiled_metrics.update_state(y_train, y_predict)
        return {m.name: m.result() for m in self.metrics}

    
    def test_step(self, data):
        x_test, y_test = data
        y_predict = self(tf.repeat(tf.expand_dims(x_test, axis=2), repeats=self.n_samples_val, axis=2), training=True)
        self.compiled_loss(y_test, y_predict)
        self.compiled_metrics.update_state(y_test, y_predict)
        return {m.name: m.result() for m in self.metrics}

    
    

########## core models ##############
class GeneratorModel(tf.keras.Model):
    def __init__(self,
                 dim_out,
                 dim_latent, 
                 n_layers, 
                 n_neurons, 
                 activation, 
                 output_activation="linear",
                 train_with_softrank_layer = True,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.dim_out = dim_out
        self.dim_latent = dim_latent
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.activation = activation
        self.output_activation = output_activation
        self.train_with_softrank_layer = train_with_softrank_layer
        


class ConcatGenerator(GeneratorModel):
    def __init__(self, dim_out, dim_latent, n_layers, n_neurons, activation, **kwargs):
        super().__init__(dim_out, dim_latent, n_layers, n_neurons, activation, **kwargs)
        
        # x has input dim (bs, n_features, n_samples)
        self.sample_noise = UnconditionalGaussianSampling(dim_latent=self.dim_latent)
        self.concat_x_and_noise = layers.Concatenate(axis=1)
        self.permute_inputs = layers.Permute((2,1))
        
        self.dense_layers = []
        for i in range(self.n_layers):
            self.dense_layers.append(layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1, activation=self.activation))

        self.output_layer = layers.Conv1D(filters=self.dim_out, kernel_size=1, strides=1, activation=self.output_activation)
        if self.train_with_softrank_layer:
            self.soft_rank_layer = SoftRank()
        self.permute_outputs = layers.Permute((2,1))
        self.clip = ClipValues(low=self.censored_left, high=self.censored_right)

        
    def call(self, inputs, training):
        #unpack and prepare inputs
        x = inputs # (bs, n_features, n_samples_train)        
        z = self.sample_noise(x) # (bs, dim_latent, n_samples_train)

        # forward pass
        h = self.concat_x_and_noise([x, z]) # -> (bs, n_features+dim_latent , n_samples_train)
        h = self.permute_inputs(h) # -> (bs, n_samples_train, n_features+dim_latent)
        
        for layer_i in self.dense_layers:
            h = layer_i(h)

        y = self.output_layer(h)
        if self.train_with_softrank_layer:
            y = self.soft_rank_layer(y, training=training)
        y = self.permute_outputs(y)
        y = self.clip(y)
        
        return y


class FiLMGenerator(GeneratorModel):
    def __init__(self, dim_out, dim_latent, n_layers, n_neurons, activation,  modulation="per_layer", **kwargs):
        super().__init__(dim_out, dim_latent, n_layers, n_neurons, activation, **kwargs)

        self.modulation = modulation

        # init layers
        self.sample_noise = UnconditionalGaussianSampling(dim_latent=self.dim_latent)
        self.permute_inputs = layers.Permute((2,1))
        
        if self.modulation == "per_layer":
            self.gamma_layers = []
            self.beta_layers = []
            for i in range(self.n_layers):
                self.gamma_layers.append(layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1,  data_format="channels_last"))
                self.beta_layers.append(layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1,  data_format="channels_last"))
        
        elif self.modulation == "constant":
            self.gamma_layer = layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1,  data_format="channels_last", name="film_gamma")
            self.beta_layer = layers.Conv1D(filters=self.n_neurons, kernel_size=1, strides=1,  data_format="channels_last", name="film_beta")
        
        else:
            raise ValueError("Unknown modulation.")
        
        self.FiLM_layers = []
        for i in range(self.n_layers):
            self.FiLM_layers.append(FiLM(n_neurons=self.n_neurons, activation=self.activation))
        
        self.output_layer = layers.Conv1D(filters=self.dim_out, kernel_size=1, strides=1, activation=self.output_activation)
        if self.train_with_softrank_layer:
            self.soft_rank_layer = SoftRank()
        self.permute_outputs = layers.Permute((2,1))
        self.clip = ClipValues(low=self.censored_left, high=self.censored_right)



    def call(self, inputs, training):
        # inputs
        x = inputs # (bs, n_features, n_samples_train)
        z = self.sample_noise(x) # (bs, dim_latent, n_samples_train)
        x = self.permute_inputs(x) # (bs, n_features, n_samples) --> (bs, n_samples, n_features)
        z = self.permute_inputs(z) # (bs, dim_latent, n_samples) --> (bs, n_samples, dim_latent)
        
        # compute gamma and beta for constant modulation
        if self.modulation == "constant":
            gamma = self.gamma_layer(z) # (bs, n_samples, dim_latent) --> (bs, n_samples, n_neurons)
            beta = self.beta_layer(z) # (bs, n_samples, dim_latent) --> (bs, n_samples, n_neurons)
        
        # forward pass
        h = x
        for i in range(len(self.FiLM_layers)):
            if self.modulation == "per_layer":
                gamma = self.gamma_layers[i](z)
                beta = self.beta_layers[i](z)
            h = self.FiLM_layers[i]([h, gamma, beta]) # (bs, n_samples, n_neurons)x3 --> (bs, n_samples, n_neurons)

        y = self.output_layer(h)
        if self.train_with_softrank_layer:
            y = self.soft_rank_layer(y, training=training)
        y = self.permute_outputs(y)
        y = self.clip(y)
        
        return y


################ define soft rank layer ################
class SoftRank(layers.Layer):
    """Differentiable ranking layer"""
    def __init__(self, alpha=1000.0):
        super(SoftRank, self).__init__()
        self.alpha = alpha # constant for scaling the sigmoid to approximate sign function, larger values ensure better ranking, overflow is handled properly by tensorflow

    def call(self, inputs, training=None):
        # input is a ?xSxD tensor, we wish to rank the S samples in each dimension per each batch
        # output is  ?xSxD tensor where for each dimension the entries are (rank-0.5)/N_rank
        if training:
            x = tf.expand_dims(inputs, axis=-1) #(?,S,D) -> (?,S,D,1)
            x_2 = tf.tile(x, (1,1,1,tf.shape(x)[1])) # (?,S,D,1) -> (?,S,D,S) (samples are repeated along axis 3, i.e. the last axis)
            x_1 = tf.transpose(x_2, (0,3,2,1)) #  (?,S,D,S) -> (?,S,D,S) (samples are repeated along axis 1)
            #return tf.transpose(1.0-0.5+tf.reduce_sum(tf.sigmoid(self.alpha*(x_1-x_2)), axis=1), perm=(0,2,1))/(tf.cast(tf.shape(x)[1], dtype=tf.float32)+1.0)
            return tf.transpose(tf.reduce_sum(tf.sigmoid(self.alpha*(x_1-x_2)), axis=1), perm=(0,2,1))/(tf.cast(tf.shape(x)[1], dtype=tf.float32))
        return inputs
    
    def get_config(self):
        return {"alpha": self.alpha}