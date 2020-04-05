from covid_data import CovidData
from plot_utils import plot_confirmed, plot_deaths, plot_confirmed_timeshift, plot_deaths_timeshift
from pathlib import Path
from datetime import date

# Data files:
time_series_folder = 'csse_covid_19_data/csse_covid_19_time_series/'
time_series_confirmed_us = 'time_series_covid19_confirmed_us.csv'
time_series_confirmed_global = 'time_series_covid19_confirmed_global.csv'
time_series_deaths_us = 'time_series_covid19_deaths_us.csv'
time_series_deaths_global = 'time_series_covid19_deaths_global.csv'
time_series_recovered_global = 'time_series_covid19_recovered_global.csv'

static_folder = 'population_data/'
population_global = 'un_world_population_2019.csv'
population_us = 'us_states_census.csv'

# Data analysis parameters:
min_confirmed_rate = 1.                     # Minimum rate of confirmed cases (confirmed/1000 inhabitants)
min_death_rate = 0.05                       # Minimum rate of deaths (deaths/1000 inhabitants)
min_population = 5e6                        # Minimum population of analyzed countries/states
req_countries = ['Norway', 'Sweden', 'Denmark', 'United States of America']

covid_data = CovidData(time_series_folder=time_series_folder,
                       time_series_confirmed_us=time_series_confirmed_us,
                       time_series_confirmed_global=time_series_confirmed_global,
                       time_series_deaths_us=time_series_deaths_us,
                       time_series_deaths_global=time_series_deaths_global,
                       time_series_recovered_global=time_series_recovered_global,
                       static_folder=static_folder,
                       population_global=population_global,
                       population_us=population_us)
covid_data.build_master_dfs()


date_str = str(date.today())
folder = 'plots/' + date_str
Path(folder).mkdir(parents=True, exist_ok=True)

plot_confirmed(covid_data.covid_data_global, folder, min_confirmed_rate=min_confirmed_rate, min_population=min_population, req_countries=req_countries)
plot_deaths(covid_data.covid_data_global, folder, min_death_rate=min_death_rate, min_population=min_population, req_countries=req_countries)
plot_confirmed_timeshift(covid_data.covid_data_global, folder, min_confirmed_rate=min_confirmed_rate, min_population=min_population, req_countries=req_countries)
plot_deaths_timeshift(covid_data.covid_data_global, folder, min_death_rate=min_death_rate, min_population=min_population, req_countries=req_countries)


