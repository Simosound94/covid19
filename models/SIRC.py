
import numpy as np

class SIRC(object):
    """ Implements the SIR epidemiology model 
        (Susceptibles, Infectuos, Recovered, Casualties)

    Learned params:
        beta    = Average contact rate
        gamma   = infectuos period (time before death or recovery or isolated) 
                  (espressed as timesteps_infectuos)
        delta   = Survival rate (recovered vs deads)
    """

    def __init__(self):
        self.set_params()

    def type(self):
        return 'SIRC'


    def set_params(self, 
                beta_mu   = None, gamma_mu  = None, delta_mu = None,
                beta_rho  = None, gamma_rho = None, delta_rho = None):

        if beta_mu is None:
            beta_mu = np.random.uniform(0, 5)
        if gamma_mu is None:
            gamma_mu = np.random.uniform(10, 40)
        if beta_rho is None:
            beta_rho = np.random.uniform(0, 5)
        if gamma_rho is None:
            gamma_rho = np.random.uniform(0, 10)
        if delta_mu is None:
            delta_mu = np.random.uniform(0, 1)
        if delta_rho is None:
            delta_rho = np.random.uniform(0, 0.5)
        
        self._beta_mu   = beta_mu  
        self._gamma_mu  = gamma_mu 
        self._beta_rho  = beta_rho 
        self._gamma_rho = gamma_rho
        self._delta_mu  = delta_mu 
        self._delta_rho = delta_rho

        

    def get_params(self):
        return {
            'beta_mu' : self._beta_mu,
            'gamma_mu' : self._gamma_mu,
            'beta_rho' : self._beta_rho,
            'gamma_rho' : self._gamma_rho,
            'delta_mu' : self._delta_mu,
            'delta_rho' : self._delta_rho,}
        

    def step(self, time, SIRC):
        """ Performs a step of the SEIR model
            interface is compliant with
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
        """
        S = SIRC[0]
        I = SIRC[1]
        R = SIRC[2]
        C = SIRC[3]
        return list(self._step(S, I, R, C))


    def _step(self, S, I, R, C, params = {}):
        """ Performs a step of the SEIR model
        Args:
            S,I,R,C (floats) Susceptibles, Infectuos, Recovered, Casualties
                             at timestep t
            params (dict) containing the 'gamma' and 'beta' parameters
        Returns:
            S,I,R,C (floats) Susceptibles, Infectuos, Recovered, Casualties
                             at timestep t + 1

        """
        nr_samples = (S.shape[0],) if len(S.shape) > 0 else ()
        if 'gamma' not in params:
            # sample weights from Normal distribution
            gamma = np.random.normal(self._gamma_mu, self._gamma_rho, size=nr_samples)
        else:
            gamma = params['gamma']

        if 'beta' not in params:
            beta = np.random.normal(self._beta_mu, self._beta_rho, size=nr_samples)
        else:
            beta = params['beta']
            
        if 'delta' not in params:
            delta = np.random.normal(self._delta_mu, self._delta_rho, size=nr_samples)
        else:
            delta = params['delta']

        # we need the inverse of gamma params
        gamma = 1 / (gamma + np.finfo(np.float32).eps)

        #Compute the SIR results
        S_to_I = beta*S*I
        I_to_RC = gamma*I
        dS = -S_to_I
        dI = S_to_I - I_to_RC
        RC = I_to_RC
        dR = RC*(1-delta)
        dC = RC*delta

        S_next, I_next, R_next, C_next = S+dS, I+dI, R+dR, C+dC

        # Clip between 0 and 1 since data is supposed to be normalized
        S_next = np.clip(S_next, 0, 1)
        I_next = np.clip(I_next, 0, 1)
        R_next = np.clip(R_next, 0, 1)
        C_next = np.clip(C_next, 0, 1)
        return S_next, I_next, R_next, C_next

    def predict(self, S, I, R, C, timesteps):
        """ Predict with the SIRC model
        Args:
            S,I,R,C (floats) Susceptibles, Exposed, Infectuos, Recovered, 
                             Casualtiesies at timestep t
            timesteps (int) How many timesteps to predict
        Returns:
            S,I,R,C (lists) of length  timesteps
        """
        S_pred = []
        I_pred = []
        R_pred = []
        C_pred = []
        nr_samples = (S.shape[0],) if len(S.shape) > 0 else ()

        # Extract a different param for each sample
        beta = np.random.normal(self._beta_mu, self._beta_rho, size=nr_samples)
        gamma = np.random.normal(self._gamma_mu, self._gamma_rho, size=nr_samples)
        delta = np.random.normal(self._delta_mu, self._delta_rho, size=nr_samples)
        if nr_samples == 1:
            beta = beta[0]
            gamma = gamma[0]
            delta = delta[0]


        params = {'beta': beta, 'gamma': gamma, 'delta':delta}

        for t in range(timesteps):
            S, I, R, C = self._step(S, I, R, C, params)
            S_pred.append(S)
            I_pred.append(I)
            R_pred.append(R)
            C_pred.append(C)
        
        SIRC = np.stack([S_pred, I_pred, R_pred, C_pred]).T
        return SIRC
