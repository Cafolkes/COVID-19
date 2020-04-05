from covid_data import CovidData
from plot_utils import plot_confirmed, plot_deaths

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
min_confirmed_rate = 1.                 # Minimum rate of confirmed cases (confirmed/1000 inhabitants)
min_death_rate = 0.1                    # Minimum rate of deaths (deaths/1000 inhabitants)

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

plot_confirmed(covid_data.covid_data_global, min_confirmed_rate=min_confirmed_rate)
plot_deaths(covid_data.covid_data_global, min_death_rate=min_death_rate)


