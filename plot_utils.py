import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

def plot_time_series(covid_data_df, folder, min_confirmed_rate, min_fatal_rate, min_population, req_areas=[], global_cases=True):
    plot_confirmed(covid_data_df, folder, min_confirmed_rate=min_confirmed_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_confirmed_norm(covid_data_df, folder, min_confirmed_rate=min_confirmed_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_confirmed_timeshift(covid_data_df, folder, min_confirmed_rate=min_confirmed_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_fatal(covid_data_df, folder, min_fatal_rate=min_fatal_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_fatal_norm(covid_data_df, folder, min_fatal_rate=min_fatal_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_fatal_timeshift(covid_data_df, folder, min_fatal_rate=min_fatal_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)

def plot_daily(covid_data_df, folder, min_confirmed_rate, min_fatal_rate, min_population, req_areas=[], global_cases=True):
    plot_confirmed_daily(covid_data_df, folder, min_confirmed_rate=min_confirmed_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_confirmed_daily_norm(covid_data_df, folder, min_confirmed_rate=min_confirmed_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_confirmed_daily_norm_timeshift(covid_data_df, folder, min_confirmed_rate=min_confirmed_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_fatal_daily(covid_data_df, folder, min_fatal_rate=min_fatal_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_fatal_daily_norm(covid_data_df, folder, min_fatal_rate=min_fatal_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)
    plot_fatal_daily_norm_timeshift(covid_data_df, folder, min_fatal_rate=min_fatal_rate, min_population=min_population, req_areas=req_areas, global_cases=global_cases)

def prepare_timeseries_data(index, rate, population, covid_df, req_areas=[], global_cases=True):
    plot_df = covid_df.loc[covid_df[index] >= rate]
    plot_df = plot_df.loc[plot_df['Population'] >= population / 1e3]

    # Add required countries back to dataframe:
    for c in req_areas:
        if c not in plot_df.index.tolist():
            plot_df = plot_df.append(covid_df.loc[c])

    if global_cases:
        plot_df.drop(['Population', 'PopulationDensity'], axis='columns', inplace=True)
    else:
        plot_df.drop(['Population'], axis='columns', inplace=True)
    plot_df.columns = pd.MultiIndex.from_tuples(plot_df.columns)

    date_lst = np.array([d[0] for d in plot_df.columns.tolist()])
    date_lst = np.unique(date_lst).tolist()
    dates = np.array([datetime.strptime(x, '%m/%d/%y') for x in date_lst])
    x_plot = np.sort(dates)

    return x_plot, plot_df

def prepare_daily_data(index, rate, population, covid_df, req_areas=[], global_cases=True, normalize=False):
    plot_df = covid_df.loc[covid_df[index] >= rate]
    plot_df = plot_df.loc[plot_df['Population'] >= population / 1e3]

    # Add required countries back to dataframe:
    for c in req_areas:
        if c not in plot_df.index.tolist():
            plot_df = plot_df.append(covid_df.loc[c])

    if global_cases:
        plot_df.drop(['Population', 'PopulationDensity'], axis='columns', inplace=True)
    else:
        plot_df.drop(['Population'], axis='columns', inplace=True)

    plot_df.columns = pd.MultiIndex.from_tuples(plot_df.columns)

    country_lst = plot_df.index.tolist()
    if global_cases:
        confirmed_daily_arr = np.empty((len(country_lst), int(plot_df.shape[1]/6)))
        fatal_daily_arr = np.empty((len(country_lst), int(plot_df.shape[1]/6)))
        recovered_daily_arr = np.empty((len(country_lst), int(plot_df.shape[1]/6)))
    else:
        confirmed_daily_arr = np.empty((len(country_lst), int(plot_df.shape[1]/4)))
        fatal_daily_arr = np.empty((len(country_lst), int(plot_df.shape[1]/4)))

    for ii, c in enumerate(country_lst):
        if normalize:
            confirmed_arr_tmp = np.array(plot_df.loc[c, (slice(None), 'confirmed, norm')].tolist())
            fatal_arr_tmp = np.array(plot_df.loc[c, (slice(None), 'fatal, norm')].tolist())
            if global_cases:
                recovered_arr_tmp = np.array(plot_df.loc[c, (slice(None), 'recovered, norm')].tolist())
        else:
            confirmed_arr_tmp = np.array(plot_df.loc[c, (slice(None), 'confirmed')].tolist())
            fatal_arr_tmp = np.array(plot_df.loc[c, (slice(None), 'fatal')].tolist())
            if global_cases:
                recovered_arr_tmp = np.array(plot_df.loc[c, (slice(None), 'recovered')].tolist())

        confirmed_daily_arr_tmp = np.diff(confirmed_arr_tmp)
        confirmed_daily_arr[ii,:] = np.concatenate((np.zeros((1,)),confirmed_daily_arr_tmp))

        fatal_daily_arr_tmp = np.diff(fatal_arr_tmp)
        fatal_daily_arr[ii,:] = np.concatenate((np.zeros((1,)), fatal_daily_arr_tmp))

        if global_cases:
            recovered_daily_arr_tmp = np.diff(recovered_arr_tmp)
            recovered_daily_arr[ii,:] = np.concatenate((np.zeros((1,)), recovered_daily_arr_tmp))

    col_lst = plot_df.columns.tolist()
    jj = 0
    cur_date = []
    cur_date.append(col_lst[0][0])

    for d in col_lst:
        if d[0] not in cur_date:
            jj += 1
            cur_date.append(d[0])
            plot_df[d[0], 'confirmed, daily'] = confirmed_daily_arr[:, jj]
            plot_df[d[0], 'fatal, daily'] = fatal_daily_arr[:, jj]
            if global_cases:
                plot_df[d[0], 'recovered, daily'] = recovered_daily_arr[:, jj]

    date_lst = np.array([d[0] for d in plot_df.columns.tolist()])
    date_lst = np.unique(date_lst).tolist()
    dates = np.array([datetime.strptime(x, '%m/%d/%y') for x in date_lst])
    x_plot = np.sort(dates)

    return x_plot, plot_df

def plot_confirmed(covid_df, folder, min_confirmed_rate=0., min_population=0, req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-3]
    else:
        index = covid_df.columns.tolist()[-2]
    x_plot, plot_df = prepare_timeseries_data(index, min_confirmed_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19,12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'confirmed')].tolist())
        start_ind = np.min(np.where(y_plot >= 0.01))
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plot[start_ind:], y_plot[start_ind:], lw=lw, label=c)
    plt.xlabel('Date')
    plt.ylabel('Confirmed cases')
    plt.title('Confirmed cases, countries with more than ' + str(min_confirmed_rate) + ' cases per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/confirmed.pdf', format='pdf', dpi='800')
    plt.close()

def plot_confirmed_norm(covid_df, folder, min_confirmed_rate=0., min_population=0, req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-3]
    else:
        index = covid_df.columns.tolist()[-2]
    x_plot, plot_df = prepare_timeseries_data(index, min_confirmed_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19,12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'confirmed, norm')].tolist())
        start_ind = np.min(np.where(y_plot >= 0.01))
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plot[start_ind:], y_plot[start_ind:], lw=lw, label=c)
    plt.xlabel('Date')
    plt.ylabel('Confirmed cases per 1000 inhabitants')
    plt.title('Confirmed cases per 1000 inhabitants, countries with more than ' + str(min_confirmed_rate) + ' cases per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/confirmed_normalized.pdf', format='pdf', dpi='800')
    plt.close()

def plot_confirmed_timeshift(covid_df, folder, min_confirmed_rate=0., min_population=0, req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-3]
    else:
        index = covid_df.columns.tolist()[-2]
    x_plot, plot_df = prepare_timeseries_data(index, min_confirmed_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases)
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
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plt, y_plt, lw=lw, label=c + ', day 0 = ' + str(x[ii][0].date()))
    plt.xlabel('Days after day 0')
    plt.ylabel('Confirmed cases per 1000 inhabitants')
    plt.title('Confirmed cases per 1000 inhabitants, countries with more than ' + str(min_confirmed_rate) + ' cases per 1000 inhabitants, timeshifted')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/confirmed_normalized_timeshifted.pdf', format='pdf', dpi='800')
    plt.close()

def plot_fatal(covid_df, folder, min_fatal_rate=0., min_population=0., req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-2]
    else:
        index = covid_df.columns.tolist()[-1]
    x_plot, plot_df = prepare_timeseries_data(index, min_fatal_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19, 12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'fatal')].tolist())
        try:
            start_ind = np.min(np.where(y_plot >= 0.01))
        except:
            start_ind = 0
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plot[start_ind:], y_plot[start_ind:], lw=lw, label=c)
    plt.xlabel('Date')
    plt.ylabel('Fatalities per 1000 inhabitants')
    plt.title('Fatal cases, countries with more than ' + str(min_fatal_rate) + 'fatalities per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/fatal.pdf', format='pdf', dpi='800')
    plt.close()

def plot_fatal_norm(covid_df, folder, min_fatal_rate=0., min_population=0., req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-2]
    else:
        index = covid_df.columns.tolist()[-1]
    x_plot, plot_df = prepare_timeseries_data(index, min_fatal_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19, 12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'fatal, norm')].tolist())
        try:
            start_ind = np.min(np.where(y_plot >= 0.01))
        except:
            start_ind = 0
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plot[start_ind:], y_plot[start_ind:], lw=lw, label=c)
    plt.xlabel('Date')
    plt.ylabel('Fatalities per 1000 inhabitants')
    plt.title('Fatal cases per 1000 inhabitants, countries with more than ' + str(min_fatal_rate) + 'fatalities per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/fatal_normalized.pdf', format='pdf', dpi='800')
    plt.close()

def plot_fatal_timeshift(covid_df, folder, min_fatal_rate=0., min_population=0., req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-2]
    else:
        index = covid_df.columns.tolist()[-1]
    x_plot, plot_df = prepare_timeseries_data(index, min_fatal_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases)
    country_lst = plot_df.index.tolist()

    x, y = [], []
    for ii, c in enumerate(country_lst):
        y_plot = np.array(plot_df.loc[c, (slice(None), 'fatal, norm')].tolist())
        try:
            start_ind = np.min(np.where(y_plot >= 0.01))
        except:
            start_ind = 0
        x.append(x_plot[start_ind:])
        y.append(y_plot[start_ind:])

    plt.figure(figsize=(19, 12))
    for ii, c in enumerate(country_lst):
        x_plt = [ii for ii in range(len(x[ii]))]
        y_plt = y[ii]
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plt, y_plt, lw=lw, label=c + ', day 0 = ' + str(x[ii][0].date()))
    plt.xlabel('Days after day 0')
    plt.ylabel('Fatalities per 1000 inhabitants')
    plt.title('Fatal cases per 1000 inhabitants, countries with more than ' + str(min_fatal_rate) + 'fatalities per 1000 inhabitants, timeshifted')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/fatal_normalized_timeshifted.pdf', format='pdf', dpi='800')
    plt.close()

def plot_confirmed_daily(covid_df, folder, min_confirmed_rate=0., min_population=0, req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-3]
    else:
        index = covid_df.columns.tolist()[-2]
    x_plot, plot_df = prepare_daily_data(index, min_confirmed_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19,12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'confirmed, daily')].tolist())
        start_ind = np.min(np.where(y_plot >= 10))
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plot[start_ind+1:], y_plot[start_ind:], lw=lw, label=c)
    plt.xlabel('Date')
    plt.ylabel('Daily new cases')
    plt.title('Daily new cases, countries with more than ' + str(min_confirmed_rate) + ' cases per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/confirmed_daily.pdf', format='pdf', dpi='800')
    plt.close()

def plot_confirmed_daily_norm(covid_df, folder, min_confirmed_rate=0., min_population=0, req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-3]
    else:
        index = covid_df.columns.tolist()[-2]
    x_plot, plot_df = prepare_daily_data(index, min_confirmed_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases, normalize=True)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19,12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'confirmed, daily')].tolist())
        start_ind = np.min(np.where(y_plot >= 0.01))
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plot[start_ind+1:], y_plot[start_ind:], lw=lw, label=c)
    plt.xlabel('Date')
    plt.ylabel('New cases per 1000 inhabitants')
    plt.title('Daily new cases per 1000 inhabitants, countries with more than ' + str(min_confirmed_rate) + ' cases per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/confirmed_daily_normalized.pdf', format='pdf', dpi='800')
    plt.close()

def plot_confirmed_daily_norm_timeshift(covid_df, folder, min_confirmed_rate=0., min_population=0, req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-3]
    else:
        index = covid_df.columns.tolist()[-2]
    x_plot, plot_df = prepare_daily_data(index, min_confirmed_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases, normalize=True)
    country_lst = plot_df.index.tolist()
    x, y = [], []
    for ii, c in enumerate(country_lst):
        y_plot = np.array(plot_df.loc[c, (slice(None), 'confirmed, daily')].tolist())
        start_ind = np.min(np.where(y_plot >= 0.01))
        x.append(x_plot[start_ind+1:])
        y.append(y_plot[start_ind:])

    plt.figure(figsize=(19, 12))
    for ii, c in enumerate(country_lst):
        x_plt = [ii for ii in range(len(x[ii]))]
        y_plt = y[ii]
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plt, y_plt, lw=lw, label=c + ', day 0 = ' + str(x[ii][0].date()))

    plt.xlabel('Date')
    plt.ylabel('New cases per 1000 inhabitants')
    plt.title('Daily new cases per 1000 inhabitants, countries with more than ' + str(min_confirmed_rate) + ' cases per 1000 inhabitants, timeshifted')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/confirmed_daily_normalized_timeshifted.pdf', format='pdf', dpi='800')
    plt.close()

def plot_fatal_daily(covid_df, folder, min_fatal_rate=0., min_population=0, req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-2]
    else:
        index = covid_df.columns.tolist()[-1]
    x_plot, plot_df = prepare_daily_data(index, min_fatal_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19,12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'fatal, daily')].tolist())
        try:
            start_ind = np.min(np.where(y_plot >= 0.0001))
        except:
            start_ind = 0
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plot[start_ind+1:], y_plot[start_ind:], lw=lw, label=c)
    plt.xlabel('Date')
    plt.ylabel('Daily new fatalities')
    plt.title('Daily new fatal cases, countries with more than ' + str(min_fatal_rate) + ' fatalities per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/fatal_daily.pdf', format='pdf', dpi='800')
    plt.close()

def plot_fatal_daily_norm(covid_df, folder, min_fatal_rate=0., min_population=0, req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-2]
    else:
        index = covid_df.columns.tolist()[-1]
    x_plot, plot_df = prepare_daily_data(index, min_fatal_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases, normalize=True)
    country_lst = plot_df.index.tolist()

    plt.figure(figsize=(19,12))
    for c in country_lst:
        y_plot = np.array(plot_df.loc[c, (slice(None), 'fatal, daily')].tolist())
        try:
            start_ind = np.min(np.where(y_plot >= 0.0001))
        except:
            start_ind = 0
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plot[start_ind+1:], y_plot[start_ind:], lw=lw, label=c)
    plt.xlabel('Date')
    plt.ylabel('Daily new fatalities per 1000 inhabitants')
    plt.title('Daily new fatal cases per 1000 inhabitants, countries with more than ' + str(min_fatal_rate) + ' fatalities per 1000 inhabitants')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/fatal_daily_normalized.pdf', format='pdf', dpi='800')
    plt.close()

def plot_fatal_daily_norm_timeshift(covid_df, folder, min_fatal_rate=0., min_population=0, req_areas=[], global_cases=True):
    if global_cases:
        index = covid_df.columns.tolist()[-2]
    else:
        index = covid_df.columns.tolist()[-1]
    x_plot, plot_df = prepare_daily_data(index, min_fatal_rate, min_population, covid_df, req_areas=req_areas, global_cases=global_cases, normalize=True)
    country_lst = plot_df.index.tolist()
    x, y = [], []
    for ii, c in enumerate(country_lst):
        y_plot = np.array(plot_df.loc[c, (slice(None), 'fatal, daily')].tolist())
        try:
            start_ind = np.min(np.where(y_plot >= 0.0001))
        except:
            start_ind = 0
        x.append(x_plot[start_ind + 1:])
        y.append(y_plot[start_ind:])

    plt.figure(figsize=(19, 12))
    for ii, c in enumerate(country_lst):
        x_plt = [ii for ii in range(len(x[ii]))]
        y_plt = y[ii]
        if c in req_areas:
            lw = 3
        else:
            lw = 1.0
        plt.plot(x_plt, y_plt, lw=lw, label=c + ', day 0 = ' + str(x[ii][0].date()))

    plt.xlabel('Date')
    plt.ylabel('Daily new fatalities per 1000 inhabitants')
    plt.title('Daily new fatal cases per 1000 inhabitants, countries with more than ' + str(min_fatal_rate) + ' fatalities per 1000 inhabitants, timeshifted')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folder + '/fatal_daily_normalized_timeshifted.pdf', format='pdf', dpi='800')
    plt.close()