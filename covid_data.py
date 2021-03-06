import pandas as pd

class CovidData:
    def __init__(self, time_series_folder, time_series_confirmed_us, time_series_confirmed_global,
                 time_series_fatal_us, time_series_fatal_global, time_series_recovered_global,
                 static_folder, population_global, population_us):

        self.confirmed_us_df = pd.read_csv(time_series_folder + time_series_confirmed_us)
        self.confirmed_global_df = pd.read_csv(time_series_folder + time_series_confirmed_global)
        self.fatal_us_df = pd.read_csv(time_series_folder + time_series_fatal_us)
        self.fatal_global_df = pd.read_csv(time_series_folder + time_series_fatal_global)
        self.recovered_global_df = pd.read_csv(time_series_folder + time_series_recovered_global)
        self.population_us_df = pd.read_csv(static_folder + population_us, index_col=0)
        self.population_global_df = pd.read_csv(static_folder + population_global)

        self.covid_data_global = None
        self.covid_data_us = None

    def build_master_dfs(self):

        self._build_global_master_df()
        self._build_us_master_df()
        self._add_normalized_data()

    def _build_global_master_df(self):
        # - Combine confirmed, fatal, and recovered in single dataframe:
        self.covid_data_global = self.confirmed_global_df.copy()
        col_lst = self.covid_data_global.columns.tolist()[4:]
        for d in col_lst:
            self.covid_data_global[d, 'confirmed'] = self.confirmed_global_df[d]
            self.covid_data_global[d, 'fatal'] = self.fatal_global_df[d]
            self.covid_data_global[d, 'recovered'] = self.recovered_global_df[d]
        self.covid_data_global.drop(col_lst, axis='columns', inplace=True)

        # - Remove unnecessary columns and aggregate regions:
        self.covid_data_global.drop(['Lat', 'Long'], axis='columns', inplace=True)
        self.covid_data_global = self.covid_data_global.groupby(self.covid_data_global['Country/Region']).aggregate('sum')

        # - Align labeling of various data sources:
        self.covid_data_global.rename(index={'US': 'United States of America', 'Korea, South':'Republic of Korea',
                                             'Iran': 'Iran (Islamic Republic of)', 'Russia': 'Russian Federation',
                                             'Venezuela': 'Venezuela (Bolivarian Republic of)',
                                             'Tanzania': 'United Republic of Tanzania',
                                             'Taiwan*': 'China, Taiwan Province of China',
                                             'Syria': 'Syrian Arab Republic', 'Moldova': 'Republic of Moldova',
                                             }, inplace=True)

        # - Add population and population density column:
        population_global_red = self.population_global_df.loc[self.population_global_df['Time'] == 2019]
        population, density, missing_pop = [], [], []
        for country in self.covid_data_global.index.tolist():
            pop_tmp = population_global_red.loc[population_global_red['Location'] == country]['PopTotal'].values
            dens_tmp = population_global_red.loc[population_global_red['Location'] == country]['PopDensity'].values
            if len(pop_tmp) == 1 and len(dens_tmp) == 1:
                population.append(pop_tmp[0])
                density.append(dens_tmp[0])
            else:
                missing_pop.append(country)

        self.covid_data_global.drop(missing_pop, axis='rows', inplace=True)  # Remove countries with missing data
        self.covid_data_global['Population'] = population
        self.covid_data_global['PopulationDensity'] = density

    def _build_us_master_df(self):
        # - Combine confirmed, fatal, and recovered in single dataframe:
        self.covid_data_us = self.confirmed_us_df.copy()
        col_lst = self.covid_data_us.columns.tolist()[11:]
        for d in col_lst:
            self.covid_data_us[d, 'confirmed'] = self.confirmed_us_df[d]
            self.covid_data_us[d, 'fatal'] = self.fatal_us_df[d]
        self.covid_data_us.drop(col_lst, axis='columns', inplace=True)

        # - Remove unnecessary columns and aggregate regions:
        self.covid_data_us.drop(['UID', 'iso2', 'iso3','code3', 'FIPS', 'Country_Region', 'Lat', 'Long_', 'Combined_Key'],
                                axis='columns', inplace=True)
        self.covid_data_us = self.covid_data_us.groupby(self.covid_data_us['Province_State']).aggregate(
            'sum')

        # - Add population and population density column:
        population_us_red = self.population_us_df[['2019']]
        population_us_red = population_us_red.dropna()
        index_change = {}
        for i in population_us_red.index.tolist():
            if i[0] == '.':
                index_change[i] = str(i[1:])

        population_us_red.rename(index=index_change, inplace=True)
        population, missing_pop = [], []
        for state in self.covid_data_us.index.tolist():
            try:
                pop_tmp = population_us_red.loc[state].values
                population.append(float(pop_tmp[0].replace(',',''))/1e3)
            except:
                missing_pop.append(state)
        self.covid_data_us.drop(missing_pop, axis='rows', inplace=True)  # Remove countries with missing data
        self.covid_data_us['Population'] = population

    def _add_normalized_data(self):
        col_lst = self.covid_data_global.columns.tolist()[:-2]
        for d in col_lst:
            self.covid_data_global[d[0], d[1] + ', norm'] = self.covid_data_global[d] / self.covid_data_global['Population']

        col_lst = self.covid_data_us.columns.tolist()[:-1]
        for d in col_lst:
            self.covid_data_us[d[0], d[1] + ', norm'] = self.covid_data_us[d] / self.covid_data_us[
                'Population']
