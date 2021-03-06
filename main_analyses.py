from covid_data import CovidData
from plot_utils import plot_time_series, plot_daily
from pathlib import Path
from datetime import date

# Data files:
time_series_folder = 'csse_covid_19_data/csse_covid_19_time_series/'
time_series_confirmed_us = 'time_series_covid19_confirmed_us.csv'
time_series_confirmed_global = 'time_series_covid19_confirmed_global.csv'
time_series_fatal_us = 'time_series_covid19_deaths_us.csv'
time_series_fatal_global = 'time_series_covid19_deaths_global.csv'
time_series_recovered_global = 'time_series_covid19_recovered_global.csv'

static_folder = 'population_data/'
population_global = 'un_world_population_2019.csv'
population_us = 'us_states_census.csv'

# Data analysis parameters:
min_confirmed_rate = 5.                     # Minimum rate of confirmed cases (confirmed/1000 inhabitants)
min_fatal_rate = 0.5                        # Minimum rate of fatal (fatal/1000 inhabitants)
min_confirmed_rate_us = 5.                     # Minimum rate of confirmed cases (confirmed/1000 inhabitants)
min_fatal_rate_us = 0.5                        # Minimum rate of fatal (fatal/1000 inhabitants)
min_population = 1e6                        # Minimum population of analyzed countries/states
req_countries = ['Norway', 'Sweden', 'Denmark', 'United States of America']
req_states = ['California', 'Washington', 'New York']
mov_avg_horizon = 7                           # Number of days to average in moving average

covid_data = CovidData(time_series_folder=time_series_folder,
                       time_series_confirmed_us=time_series_confirmed_us,
                       time_series_confirmed_global=time_series_confirmed_global,
                       time_series_fatal_us=time_series_fatal_us,
                       time_series_fatal_global=time_series_fatal_global,
                       time_series_recovered_global=time_series_recovered_global,
                       static_folder=static_folder,
                       population_global=population_global,
                       population_us=population_us)
covid_data.build_master_dfs()

date_str = str(date.today())
folder_global_accu = 'plots/' + date_str + '/global/accumulated'
folder_global_daily = 'plots/' + date_str + '/global/daily'
folder_us_accu = 'plots/' + date_str + '/us/accumulated'
folder_us_daily = 'plots/' + date_str + '/us/daily'
Path(folder_global_accu).mkdir(parents=True, exist_ok=True)
Path(folder_global_daily).mkdir(parents=True, exist_ok=True)
Path(folder_us_accu).mkdir(parents=True, exist_ok=True)
Path(folder_us_daily).mkdir(parents=True, exist_ok=True)

# Plot global cases:
plot_time_series(covid_data.covid_data_global, folder_global_accu, min_confirmed_rate=min_confirmed_rate, min_fatal_rate=min_fatal_rate, min_population=min_population, req_areas=req_countries)
plot_daily(covid_data.covid_data_global, folder_global_daily, min_confirmed_rate=min_confirmed_rate, min_fatal_rate=min_fatal_rate, min_population=min_population, req_areas=req_countries, mov_avg_horizon=mov_avg_horizon)

# Plot us cases:
plot_time_series(covid_data.covid_data_us, folder_us_accu, min_confirmed_rate=min_confirmed_rate_us, min_fatal_rate=min_fatal_rate_us, min_population=min_population, req_areas=req_states, global_cases=False)
plot_daily(covid_data.covid_data_us, folder_us_daily, min_confirmed_rate=min_confirmed_rate_us, min_fatal_rate=min_fatal_rate_us, min_population=min_population, req_areas=req_states, global_cases=False, mov_avg_horizon=mov_avg_horizon)
