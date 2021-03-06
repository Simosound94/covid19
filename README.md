
# Estimation of parameters of the COVID-19 (SIR model)

This repository allow estimation of parameters of the SIR and SEIR epidemic
models.

## Dependencies
Firstly, make sure all the dependencies are installed
```pip install --user --upgrade -r requirements.txt```
Download the COVID-19 data in the current repository
```git clone https://github.com/CSSEGISandData/COVID-19```


## COVID-19 params esitimation
Then the parameters of the COVID-19 can be estimated as
```python train_random_search.py```
estimates the parameters on all the countries together,
with the assumption of fixed population, equal for each country

To estimate  the parameters for a single country (e.g. Italy)
```python train_random_search.py --train_on_country=Italy --population=60000000```
You will see something similar to

![Italy](/results/Italy.png)

The repo allows also to estimate the parameters ot the epidemic models day by day
```python estimate_params_day_by_day.py --train_on_country=Italy --population=60000000 --train_iters=30000 --predicted_days=60```
You will see something similar to

The parameters of the SIRC model on italy, the last `--predicted_days` are inferred through a linear regression of the last values
![Params of Italy](/results/Italy_params.png)

The Infectuos / Recovered and Casualties, considering the parameters of the previous plot
(note that the parameters vary each day)
![Italy with day by day params](/results/Italy_day_by_day_params.png)