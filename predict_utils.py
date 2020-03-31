

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


from data import data


def reject_outliers(data, m=1.):
        """ Sets the value of outlayers to nan
        Args:
            data (np.array): the data from which to remove outlayers
            m (float): threshold on outlayers as m * std_dev(data)
        """

        standardized_data = np.abs(data - np.mean(data, axis=0))
        outlayer_threshold = m * np.std(data, axis=0)
        outlayers = standardized_data > outlayer_threshold

        # data[ outlayers ] = standardized_data[outlayers]
        data[ outlayers ] = np.nan

        print('nr of outlayers: ', np.sum(outlayers))
        return data



def predict(model, country, population, days_to_predict, params = None, std_dev=1.):
    """ Use the epidemic model to predict what's next
    Args:
        model: one of the epidemic models
        country: (str) which country to predict
        population: (int) population of the country
        days_to_predict: (int) nr of days to predict
        params: (list) [OPTIONAL] containing the params of the model for each
                      timestep to predict
        std_dev (float) standard deviation of uncertainty to plot in graph
    """

    X, _, dates = data.getDataset(
                           init_population = population,
                           type=model.type(), country=country)
    X = X[0,:,:] # first sample
    # get the last day
    S = X[-1, 0]
    I = X[-1, 1]
    R = X[-1, 2]
    C = X[-1, 3]

    # Average more predictions (models have bayesian params)
    # TODO: this procedure can be vectorized
    y_pred_accumulator = []
    for _ in range(1000):

        if params == None: # use default params of the model
            y_pred = model.predict(S, I, R, C, timesteps=days_to_predict)

        else: # Use different params, for each day
            assert len(params) == days_to_predict
            SIRC = [S, I, R, C]
            y_pred = []
            for d in range(days_to_predict):
                model.set_params(**params[d])
                SIRC = model.step(None, SIRC)
                y_pred.append(SIRC)
            y_pred = np.array(y_pred)

        y_pred_accumulator.append(y_pred)
    
    y_pred_accumulator = np.array(y_pred_accumulator)
    y_pred_accumulator = reject_outliers(y_pred_accumulator, m=std_dev)

    # Max, Min, Mean ignoring NaNs (--> outlayers)
    worst_case = np.nanmax(y_pred_accumulator, axis=0)
    best_case = np.nanmin(y_pred_accumulator, axis=0)
    average_case = np.nanmean(y_pred_accumulator, axis=0)

    timeseries_best = np.concatenate([X, best_case], axis=0)
    timeseries_worst = np.concatenate([X, worst_case], axis=0)
    timeseries_average = np.concatenate([X, average_case], axis=0)

    # Adding the population to the values
    timeseries_best *= population
    timeseries_worst *= population
    timeseries_average *= population


    # add predicted values to dates
    current = dates[-1]
    for i in range(days_to_predict):
        dates.append( current + timedelta(days = i+1))
    dates = [d.strftime('%d/%m/%y') for d in dates]

    plt.figure(figsize=(15,10))
    plt.title('Italy - COVID-19')

    # Plot average
    plt.plot(timeseries_average[:,1], label='Infected',  color='orange')
    plt.plot(timeseries_average[:,2], label='Recovered',  color='g')
    plt.plot(timeseries_average[:,3], label='Casualties',  color='r')


    # "Confidence interval"
    x = np.arange(0, len(timeseries_best)) 
    plt.fill_between(x, timeseries_best[:,1], timeseries_worst[:,1], color='orange', alpha=0.2)
    plt.fill_between(x, timeseries_best[:,2], timeseries_worst[:,2], color='g', alpha=0.2)
    plt.fill_between(x, timeseries_best[:,3], timeseries_worst[:,3], color='r', alpha=0.2)
    
    plt.xticks(x[::5], dates[::5], rotation=90)
    plt.grid(True)
    plt.legend()
    plt.plot()
    plt.savefig(str(country)+".png")
    plt.show()

    
        
