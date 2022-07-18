import numpy as np

from ..mvpreg.mvpreg.models.copulas import GaussianCopula, IndependenceCopula, SchaakeShuffle, VineCopula
from ..mvpreg.mvpreg.models.helpers import rank_data_random_tiebreaker
from ..mvpreg.mvpreg.models.parametric_regression import DeepParametricRegression

from .igc_models import ScoringRuleCopulaDGR as CIGC

class DeepParametricRegressionEnhanced(DeepParametricRegression):
    def __init__(self,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        
        self.model = self._build_model()
        if self.show_model_summary:
            self.model.summary(expand_nested=True)


    def fit_copula(self, x, y):
            pseudo_obs = self.cdf(x, y) # obtain pseudo observations
            
            # ensure that marginals of pseudo_obs are uniform
            self.pseudo_obs_uniform = np.zeros_like(pseudo_obs)
            for d in range(pseudo_obs.shape[1]):
                self.pseudo_obs_uniform[:,d] = (2*rank_data_random_tiebreaker(pseudo_obs[:,d])-1)/(2*pseudo_obs.shape[0]) # assign ordinal ranks and break ties at random
            
            if self.copula_type == "cigc":
                self.copula = CIGC(**copula_kwargs).fit(x,u)

            else:
                self.copula = self._get_copula(self.pseudo_obs_uniform) # fit copula model on uniform pobs
            
            return self


    def _get_copula(self, u, copula_kwargs):
        
        u = np.clip(u, 1e-8, 1.0-1e-8)
        
        if self.copula_type == "gaussian":
            copula = GaussianCopula().fit(u)
        
        elif self.copula_type == "independence":
            copula = IndependenceCopula(dim=u.shape[1])
            
        elif self.copula_type == "schaake":
            copula = SchaakeShuffle().fit(u)
        
        elif self.copula_type in ("r-vine", "d-vine", "c-vine"):
            copula = VineCopula(pair_copula_families=self.pair_copula_families, vine_structure=self.vine_structure, vine_type=self.copula_type).fit(u)

        #elif self.copula_type == "igc":
        #   copula = IGC(**copula_kwargs).fit(u)
        
        else:
            raise ValueError(f"Copula type {self.copula_type} is unknown. Must be one of: independence, schaake, gaussian, r-vine, c-vine, d-vine.")
        
        return copula
    
    
    def simulate(self, x, n_samples=1):
        if self.copula_type == "independence":
            return self.simulate_marginals(x, n_samples)        
        else:
            p_pred = self.predict_distributions(x)
            y_pred = []
            if self.copula_type != "cigc":
                for i in range(n_samples):
                    u = self.simulate_copula(n_samples=p_pred.shape[0]) # possible because copulas is unconditional on x 
                    y_pred.append(p_pred.quantile(u))            
                return self._rescale_y_samples(np.stack(y_pred, axis=2))
            else:
                for i in range(n_samples):
                    u = self.copula.simulate(x, n_samples=1) # possible because copulas is unconditional on x 
                    y_pred.append(p_pred.quantile(u))            
                return self._rescale_y_samples(np.stack(y_pred, axis=2))
            
    
    def simulate_copula(self, n_samples=1):
        return np.clip(self.copula.simulate(n_samples), 0.0001, 0.9999)
