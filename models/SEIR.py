import numpy as np



class SEIR(object):
    """ Implements the SEIR epidemiology model 
        (Susceptibles, Exposed, Infectuos, Recovered)

    Learned params:
        alpha   = Incubation period (expressed as timesteps_incubation)
        beta    = Average contact rate
        gamma   = infectuos period (time before death or recovery or isolated) 
                  (espressed as timesteps_infectuos)
    """

    def __init__(self):
        self.set_params()


    def set_params(self, 
                alpha_mu  = None,
                beta_mu   = None,
                gamma_mu  = None,
                alpha_rho = None,
                beta_rho  = None,
                gamma_rho = None):

        if alpha_mu is None:
            alpha_mu = np.random.uniform(0, 30)
        if beta_mu is None:
            beta_mu = np.random.uniform(1, 5)
        if gamma_mu is None:
            gamma_mu = np.random.uniform(0, 30)
        if alpha_rho is None:
            alpha_rho = np.random.uniform(0, 5)
        if beta_rho is None:
            beta_rho = np.random.uniform(0, 1)
        if gamma_rho is None:
            gamma_rho = np.random.uniform(0, 5)

        self._alpha_mu  = alpha_mu 
        self._beta_mu   = beta_mu  
        self._gamma_mu  = gamma_mu 
        self._alpha_rho = alpha_rho
        self._beta_rho  = beta_rho 
        self._gamma_rho = gamma_rho

        

    def get_params(self):
        return {
            'alpha_mu' : self._alpha_mu,
            'beta_mu' : self._beta_mu,
            'gamma_mu' : self._gamma_mu,
            'alpha_rho' : self._alpha_rho,
            'beta_rho' : self._beta_rho,
            'gamma_rho' : self._gamma_rho}
        


    def _step(self, S, E, I, R):
        """ Performs a step of the SEIR model
        Args:
            S,E,I,R (floats) Susceptibles, Exposed, Infectuos, Recovered 
                             at timestep t
        Returns:
            S,E,I,R (floats) Susceptibles, Exposed, Infectuos, Recovered 
                             at timestep t + 1

        """
        # sample weights from Normal distribution
        alpha = np.random.normal(self._alpha_mu, self._alpha_rho)
        beta = np.random.normal(self._beta_mu, self._beta_rho)
        gamma = np.random.normal(self._gamma_mu, self._gamma_rho)
        

        # we need the inverse of these params
        alpha = 1 / (alpha + np.finfo(np.float32).eps)
        gamma = 1 / (gamma + np.finfo(np.float32).eps)

        #Compute the SEIR results
        dS = -beta*S*I
        dE = beta*S*I - alpha*E
        dI = alpha*E - gamma*I
        dR = gamma*I
        S_next, E_next, I_next, R_next = S+dS, E+dE, I+dI, R+dR

        # Clip between 0 and 1 since data is supposed to be normalized
        S_next = np.clip(S_next, 0, 1)
        E_next = np.clip(E_next, 0, 1)
        I_next = np.clip(I_next, 0, 1)
        R_next = np.clip(R_next, 0, 1)
        return S_next, E_next, I_next, R_next

    def predict(self, S, E, I, R, timesteps):
        """ Predict with the SEIR model
        Args:
            S,E,I,R (floats) Susceptibles, Exposed, Infectuos, Recovered 
                             at timestep t
            timesteps (int) How many timesteps to predict
        Returns:it,
                    params['alpha_mu'], 
                    params['beta_mu'],
                    params['gamma_mu'], 
                    params['alpha_rho'],
                    params['beta_rho'],
                    params['gamma_rho'], l)
            S,E,I,R (lists) of length  timesteps
        """
        S_pred = []
        E_pred = []
        I_pred = []
        R_pred = []

        for _ in range(timesteps):
            S, E, I, R = self._step(S, E, I, R)
            S_pred.append(S)
            E_pred.append(E)
            I_pred.append(I)
            R_pred.append(R)
        
        SEIR = np.stack([S_pred, E_pred, I_pred, R_pred]).T
        return SEIR

