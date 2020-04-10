import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

def prepare_plotting_data(index, rate, population, covid_df, req_countries=[]):
    plot_df = covid_df.loc[covid_df[index] >= rate]
    plot_df = plot_df.loc[plot_df['Population'] >= population / 1e3]

    # Add required countries back to dataframe:
    for c in req_countries:
        if c not in plot_df.index.tolist():
            plot_df = plot_df.append(covid_df.loc[c])

    plot_df.drop(['Population', 'PopulationDensity'], axis='columns', inplace=True)
    plot_df.columns = pd.MultiIndex.from_tuples(plot_df.columns)

    date_lst = np.array([d[0] for d in plot_df.columns.tolist()])
    date_lst = np.unique(date_lst).tolist()
    dates = np.array([datetime.strptime(x, '%m/%d/%y') for x in date_lst])
    x_plot = np.sort(dates)

    return x_plot, plot_df

def plot_confirmed(covid_df, folder, min_confirmed_rate=0., min_population=0, req_countries=[]):
    index = covid_df.columns.tolist()[-3]
    x_plot, plot_df = prepare_plotting_data(index, min_confirmed_rate, min_population, covid_df, req_countries=req_countries)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19,12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'confirmed, norm')].tolist())
        start_ind = np.min(np.where(y_plot >= 0.01))
        plt.plot(x_plot[start_ind:], y_plot[start_ind:], label=c)
    plt.xlabel('Date')
    plt.ylabel('Confirmed cases (/1000 inhabitants')
    plt.title('Confirmed cases per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/confirmed_normalized.pdf', format='pdf', dpi='800')

def plot_deaths(covid_df, folder, min_death_rate=0., min_population=0., req_countries=[]):
    index = covid_df.columns.tolist()[-2]
    x_plot, plot_df = prepare_plotting_data(index, min_death_rate, min_population, covid_df, req_countries=req_countries)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19, 12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'confirmed, norm')].tolist())
        start_ind = np.min(np.where(y_plot >= 0.01))
        plt.plot(x_plot[start_ind:], y_plot[start_ind:], label=c)
    plt.xlabel('Date')
    plt.ylabel('Confirmed deaths (/1000 inhabitants')
    plt.title('Confirmed deaths per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/deaths_normalized.pdf', format='pdf', dpi='800')

def plot_confirmed_timeshift(covid_df, folder, min_confirmed_rate=0., min_population=0, req_countries=[]):
    index = covid_df.columns.tolist()[-3]
    x_plot, plot_df = prepare_plotting_data(index, min_confirmed_rate, min_population, covid_df, req_countries=req_countries)
    country_lst = plot_df.index.tolist()
    x, y = [], []
    for ii, c in enumerate(country_lst):
        y_plot = np.array(plot_df.loc[c, (slice(None), 'confirmed, norm')].tolist())
        start_ind = np.min(np.where(y_plot >= 0.01))
        x.append(x_plot[start_ind:])
        y.append(y_plot[start_ind:])

    plt.figure(figsize=(19,12))
    for ii, c in enumerate(country_lst):
        x_plt = [ii for ii in range(len(x[ii]))]
        y_plt = y[ii]
        if c in req_countries:
            lw = 3
        else:
            lw = 0.5
        plt.plot(x_plt, y_plt, lw=lw, label=c + ', day 0 = ' + str(x[ii][0].date()))
    plt.xlabel('Days after day 0')
    plt.ylabel('Confirmed cases (/1000 inhabitants')
    plt.title('Confirmed cases per 1000 inhabitants, time synchronized')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/confirmed_normalized_timeshifted.pdf', format='pdf', dpi='800')

def plot_deaths_timeshift(covid_df, folder, min_death_rate=0., min_population=0., req_countries=[]):
    index = covid_df.columns.tolist()[-2]
    x_plot, plot_df = prepare_plotting_data(index, min_death_rate, min_population, covid_df, req_countries=req_countries)
    country_lst = plot_df.index.tolist()

    x, y = [], []
    for ii, c in enumerate(country_lst):
        y_plot = np.array(plot_df.loc[c, (slice(None), 'confirmed, norm')].tolist())
        start_ind = np.min(np.where(y_plot >= 0.01))
        x.append(x_plot[start_ind:])
        y.append(y_plot[start_ind:])

    plt.figure(figsize=(19, 12))
    for ii, c in enumerate(country_lst):
        x_plt = [ii for ii in range(len(x[ii]))]
        y_plt = y[ii]
        if c in req_countries:
            lw = 3
        else:
            lw = 0.5
        plt.plot(x_plt, y_plt, lw=lw, label=c + ', day 0 = ' + str(x[ii][0].date()))
    plt.xlabel('Days after day 0')
    plt.ylabel('Confirmed cases (/1000 inhabitants')
    plt.title('Confirmed deaths per 1000 inhabitants, time synchronized')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/deaths_normalized_timeshifted.pdf', format='pdf', dpi='800')