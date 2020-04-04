class CovidData:
    def __init__(self, time_series_folder='csse_covid_19_data/csse_covid_19_time_series/',
                 time_series_confirmed_us='time_series_covid19_confirmed_us.csv',
                 time_series_confirmed_global='time_series_covid19_confirmed_global.csv',
                 time_series_deaths_us='time_series_covid19_deaths_us.csv',
                 time_series_deaths_global='time_series_covid19_deaths_global.csv',
                 time_series_recovered_global='time_series_covid19_recovered_global.csv',
                 population_global='population_global.csv',
                 population_us='population_us.csv'):

        self.time_series_folder = time_series_folder
        self.time_series_confirmed_us_fname = time_series_confirmed_us
        self.time_series_confirmed_global_fname = time_series_confirmed_global
        self.time_series_deaths_us_fname = time_series_deaths_us
        self.time_series_deaths_global_fname = time_series_deaths_global
        self.time_series_recovered_global_fname = time_series_recovered_global
        self.population_global_fname = population_global
        self.population_us_fname = population_us
        # TODO: Find data sets for global and us population numbers

    def import_time_series(self):
        pass

    def process_time_series(self):
        #TODO: Decide practical model structure - pandas
        pass


