import numpy as np
import pandas as pd
from datetime import timedelta, datetime

CONFIRMED = 'time_series_covid19_confirmed_global.csv'
DEATHS = 'time_series_covid19_deaths_global.csv'
RECOVERED = 'time_series_covid19_recovered_global.csv'
POPULATION = 'populations.csv'

def getDataset( init_population = None,
                country = None,
                type = 'SEIR',
                dataset='./COVID-19/csse_covid_19_data/csse_covid_19_time_series/',
                return_time_series = True
               ):
    """ Creates the time series dataset for the covid-19
    Args
        init_population (int) the whole population (None: load populations from file)
        country (str): select a county to return
        type (str): one of SEIR (-> returns SEIR data) or SIR (-> returns SIR data)
        dataset (str): the dataset path
        return_time_series (bool): whether to return time series as [M x T x F]
                                    (M: nr samples, T: timesteps, F: features)
                                    or [M*T x F]
    """



    population = pd.read_csv(dataset+POPULATION)
    confirmed = pd.read_csv(dataset+CONFIRMED)
    deaths = pd.read_csv(dataset+DEATHS)
    recovered = pd.read_csv(dataset+RECOVERED)

    # if not (confirmed["Country/Region"] == population["Country/Region"]).all():
    #     print("Population does not match confirmed in:")
    #     print(np.where(confirmed["Country/Region"] != population["Country/Region"]))
    #     exit()

    if country is not None:
        confirmed   = confirmed[confirmed['Country/Region'] == country]
        deaths      = deaths[deaths['Country/Region'] == country]
        recovered   = recovered[recovered['Country/Region'] == country]
        population   = population[population['Country/Region'] == country]



    confirmed = confirmed.drop(columns = ["Province/State","Country/Region","Lat","Long"])
    deaths = deaths.drop(columns = ["Province/State","Country/Region","Lat","Long"])
    recovered = recovered.drop(columns = ["Province/State","Country/Region","Lat","Long"])
    population = population.drop(columns = ["Province/State","Country/Region"])

    dates = confirmed.columns

    confirmed = confirmed.to_numpy()
    recovered = recovered.to_numpy()
    casualties = deaths.to_numpy()
    population = population.to_numpy()

    if init_population is not None:
        population = init_population



    if not (confirmed.shape == recovered.shape == casualties.shape):
        raise ValueError('Shape of confirmed {}, recovered {} and casualties {}'\
            ' should match'.format(confirmed.shape, recovered.shape, casualties.shape))

    susceptibles = np.zeros(confirmed.shape) * np.nan
    exposed = np.zeros(confirmed.shape) * np.nan
    infectous = confirmed

    if type == 'SEIR':
        # Return the SEIR data
        # NOTE: we don't have info about the exposed and the susceptibles
        # we consider at time 0 susceptibles = 1 - infectuos and exposed = 0
        # for t > 0 those data are NaN
        susceptibles[:,0] = (population - infectous[:,0] - recovered[:,0] - casualties[:,0]) / population
        exposed[:,0] = 0
        infectous = infectous / population
        recovered = recovered / population
        casualties = casualties / population

        X = np.stack((susceptibles, exposed, infectous, recovered, casualties), axis=-1)

    elif type == 'SIRC':
        infectous = infectous / population
        recovered = recovered / population
        casualties = casualties / population
        susceptibles = 1 - (infectous + recovered + casualties)
        X = np.stack((susceptibles, infectous, recovered, casualties), axis=-1)

    else:
        raise ValueError('Type {} not supported'.format(type))

    Y = X[:, 1:, :] # Y as X but starting from timestep 1, not 0 (1 timestep ahead)
    X = X[:,:-1, :]

    M, T, F = X.shape
    if not return_time_series:
        X = np.reshape(X, (M*T, F))
        Y = np.reshape(Y, (M*T, F))

        # TODO: remove zeros
        # no_zeros = np.where(np.sum(X[:,:,1:]+Y[:,:,1:], axis=(1, 2)) > 0 )[0]
        # print('removing ',len(X) - len(no_zeros), 'samples')
        # X, Y = X[no_zeros], Y[no_zeros]


    dates = [datetime.strptime(d, '%m/%d/%y') for d in dates.tolist()[1:]]
    return X, Y, dates
    

    





