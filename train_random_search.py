
import os
import numpy as np


from data import data
from models.SIRC import SIRC
from models.SEIR import SEIR
from predict_utils import predict

# ==============================================================================
#
# Constants
#
# ==============================================================================

LOSS_CV_ROUNDS = 3

# ==============================================================================
#
# Functions
#
# ==============================================================================

def mse_loss(y_true, y_pred):
    mse = (np.square(y_true - y_pred)).mean()
    return mse


def mae_loss(y_true, y_pred):
    mae = (np.absolute(y_true - y_pred)).mean()
    return mae



def main(args):
    """ Train and evaluate with a model in a NON TIME SERIES form
    """

    # ==========================================================================
    # Get the dataset
    # ==========================================================================
    X, Y, dates = data.getDataset(init_population = args.population,
                           type=args.model.upper(),
                           country=args.train_on_country,
                           return_time_series= False,
                        )

    if not np.sum(X > 1) == 0:
        raise ValueError("Data is expected normalized in [0, 1]. \
                          The selected population is not enough.")
    
    # Select the model
    if args.model.upper() == 'SIRC':
        model = SIRC()
    elif args.model.upper() == 'SEIR':
        model = SEIR()
    else:
        raise ValueError('--model not supported')

    # Select the loss
    if args.loss.upper() == 'MSE':
        loss = mse_loss
    elif args.loss.upper() == 'MAE':
        loss = mae_loss
    else:
        raise ValueError('--loss not supported')

    print("{:>15} {:>15} {:>15} {:>15} {:>15} {:>15}  {:>15}  {:>15}".format(
        'iter','beta_mu','beta_rho','gamma_mu','gamma_rho','delta_mu','delta_rho','loss'))

    # Start from last checkpoint if exists
    old_data = os.path.isfile(args.cache_file)
    f = open(args.cache_file, 'a')
    if not old_data:
        f.write("iter,beta_mu,beta_rho,gamma_mu,gamma_rho,delta_mu,delta_rho,loss\n")
        optimal_params = None
        optimal_loss = np.inf
    else:
        logs = open(args.cache_file, 'r') .read().splitlines()
        last_log = logs[-1]
        last_log = last_log.split(',')
        optimal_params = {
            'beta_mu' : float(last_log[1]),
            'beta_rho' : float(last_log[2]),
            'gamma_mu' : float(last_log[3]),
            'gamma_rho' : float(last_log[4]),
            'delta_mu' : float(last_log[5]),
            'delta_rho' : float(last_log[6]),
            }
        optimal_loss = float(last_log[7])
    
        print("{: 15d} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E}".format(
                    -1, 
                    optimal_params['beta_mu'],
                    optimal_params['beta_rho'],
                    optimal_params['gamma_mu'], 
                    optimal_params['gamma_rho'], 
                    optimal_params['delta_mu'],
                    optimal_params['delta_rho'],optimal_loss))

    # ==========================================================================
    # Train by random search
    # ==========================================================================


    y_true = Y
    timesteps = 1

    def compute_loss(params={}, X = [], y_true = [], timesteps = 1):
        # Set params (random if value is not set)
        model.set_params(**params)

        # Determine the loss with those params
        l = 0
        for _ in range(LOSS_CV_ROUNDS):
            y_pred = model.predict(X[:,0], X[:,1], X[:,2], X[:,3], timesteps)
            y_pred = np.squeeze(y_pred, axis=1) # remove timestep dimension

            l += mae_loss(y_true, y_pred)
        l /= LOSS_CV_ROUNDS
        return l
        
    
    for it in range(args.train_iters):
        if it % 1000 == 0:
            print('Iter:', it)

        l = compute_loss(X = X, y_true = y_true, timesteps = 1)
        

        params = model.get_params()
        # TODO: implement occam razor for solutions! if (opt_loss - loss) < epsilon: 
        # chose the solution with higher variance
        if l < optimal_loss:
            optimal_params = params
            optimal_loss = l

            mess = "{},{},{},{},{},{},{},{}".format(
                    it,
                    params['beta_mu'],
                    params['beta_rho'],
                    params['gamma_mu'], 
                    params['gamma_rho'],
                    params['delta_mu'],
                    params['delta_rho'], l)
            f.write(mess+"\n")
            print("{: 15d} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E} {: 15.2E}".format(
                it, 
                params['beta_mu'],
                params['beta_rho'],
                params['gamma_mu'], 
                params['gamma_rho'], 
                params['delta_mu'],
                params['delta_rho'],l))

    # ==========================================================================
    # Train by scipy minimize
    # ==========================================================================

    # from scipy import optimize

    # def compute_loss(params, X = [], y_true = [], timesteps = 1):
    #     # Set params (random if value is not set)
    #     params = {'beta_mu' : params[0],
    #             'beta_rho' : params[1],
    #             'gamma_mu' : params[2],
    #             'gamma_rho' : params[3],
    #             'delta_mu' : params[4],
    #             'delta_rho' : params[5],}

    #     model.set_params(**params)

    #     # Determine the loss with those params
    #     l = 0
    #     for _ in range(LOSS_CV_ROUNDS):
    #         y_pred = model.predict(X[:,0], X[:,1], X[:,2], X[:,3], timesteps)
    #         y_pred = np.squeeze(y_pred, axis=1) # remove timestep dimension

    #         l += mae_loss(y_true, y_pred)
    #     l /= LOSS_CV_ROUNDS
    #     return l

    # res =optimize.minimize(fun = compute_loss, 
    #                     #    x0 = np.array([0.38, 0.02, 30., 7., 0.17, 0.02]),
    #                        x0 = np.array([0.8, 0.1, 30., 1., 0.1, 0.01]),
    #                        args = (X, y_true, 1),
    #                     #    method = "L-BFGS-B",
    #                        method = "SLSQP",
    #                        # Min / max bound for beta_mu, beta_rho, ....
    #                        bounds = [(0, 5), (0, 5), (0, 40), (0, 10), (0,1), (0,1)],
    #                        tol = 1e-6,
    #                        options = {'maxiter':100000, 'disp':True}
    #                        )

        


    # ==========================================================================
    # Evaluate on Italy
    # ==========================================================================

    model.set_params(beta_mu=optimal_params['beta_mu'],
                    gamma_mu=optimal_params['gamma_mu'],
                    beta_rho=optimal_params['beta_rho'],
                    gamma_rho=optimal_params['gamma_rho'])

    COUNTRY = 'Italy'
    POPULATION = args.population
    predict(model, COUNTRY, POPULATION, args.predicted_days)
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description = "COVID-19 estimation of params")
    parser.add_argument('--loss', type=str, default='MAE', help='The loss type (MSE or MAE) [default: MSE]')
    parser.add_argument('--model', type=str, default='SIRC', help='The loss type (SIRC or SEIR) [default: SIRC]')
    parser.add_argument('--train_on_country', type=str, default=None, help='The country to use to train the model \
        (None: all the contries) [default: None]')
    parser.add_argument('--population', type=int, default=100000, help='Population [default: 100k]')
    parser.add_argument('--train_iters', type=int, default=100000, help='Train iters [default: 100k]')
    parser.add_argument('--predicted_days', type=int, default=15, help='Days to predict [default: 15]')
    parser.add_argument('--cache_file', type=str, default='optimization_steps.csv', help='The cache where to save optimal params')

    args = parser.parse_args()    

    main(args)
