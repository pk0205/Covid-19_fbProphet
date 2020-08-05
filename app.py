import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py


def main():
    st.title("COVID-19 DASHBOARD")
    st.sidebar.title("COVID-19 DASHBOARD")

    @st.cache(persist=True)
    def load_data():
        references = pd.read_csv("reference.csv")
        references.drop(['UID', 'iso2',  'code3', 'FIPS', 'Admin2', 'Province_State',
                         'Lat', 'Long_', 'Combined_Key', 'Population'], axis=1, inplace=True)
        references = references.drop_duplicates().reset_index()
        references = references.drop_duplicates("Country_Region").reset_index()
        references = references.rename(
            columns={"Country_Region": "Country/Region"})
        references.drop(['level_0', 'index'], axis=1, inplace=True)

        cov_data = pd.read_csv("time-series-19-covid-combined.csv")
        cov_data.drop(['Lat', 'Long', 'Province/State'], axis=1,  inplace=True)
        pd.to_datetime(cov_data['Date'])

        world_aggregate = pd.read_csv("worldwide-aggregated.csv")

        return references, cov_data, world_aggregate

    references, cov_data, world_aggregate = load_data()

    option = st.sidebar.selectbox(
        '',
        ('Cases till Now', 'Future Prediction'))

    if(option == 'Cases till Now'):

        dt = st.slider("Select Date", datetime.date(
            2020, 1, 22), datetime.date(2020, 7, 30), datetime.date(2020, 7, 30))

        cov_dt = cov_data[pd.to_datetime(
            cov_data['Date']) == pd.to_datetime(dt)]
        cov_dt2 = cov_dt.groupby("Country/Region").agg(
            {'Confirmed': ['sum'], 'Recovered': ['sum'], 'Deaths': ['sum']}).reset_index()

        formatted_data = pd.merge(cov_dt2, references, on='Country/Region')
        formatted_data.drop([('Country/Region', '')], axis=1, inplace=True)
        formatted_data = formatted_data.rename(
            columns={"iso3": "iso_a3"})

        plt_type = "Recovered"
        formatted_data['Confirmed'] = formatted_data[('Confirmed', 'sum')]
        formatted_data['Recovered'] = formatted_data[('Recovered', 'sum')]
        formatted_data['Deaths'] = formatted_data[('Deaths', 'sum')]
        formatted_data.drop([('Confirmed', 'sum'), ('Recovered',
                                                    'sum'), ('Deaths', 'sum')], axis=1, inplace=True)

        fig = px.choropleth(formatted_data, locations="iso_a3",
                            color="Confirmed",
                            hover_name='Country/Region',
                            color_continuous_scale=px.colors.sequential.Plasma)

        st.write(fig)

        formatted_data.sort_values(
            by='Confirmed', ascending=False, inplace=True)
        formatted_data = formatted_data.reset_index(drop=True)

        op = st.selectbox("", ['Confirmed', 'Recovered', 'Deaths'])

        total_cases = sum(formatted_data[op])
        top_7 = sum(formatted_data[op][:7])
        rest_case = total_cases-top_7
        labels = []
        sizes = []

        explode = (0.05, 0.03, 0.08, 0, 0, 0, 0, 0.05)

        for i in range(7):
            labels.append(formatted_data['Country/Region'][i])
            sizes.append(formatted_data[op][i]/total_cases)
        labels.append('Rest of World')
        sizes.append(rest_case/total_cases)
        cmap = plt.get_cmap('Spectral')
        colors = [cmap(i) for i in np.linspace(0, 1, 8)]
        plt.pie(sizes, labels=labels, explode=explode,
                autopct='%1.1f%%', shadow=True, colors=colors)

        st.pyplot()

        if st.sidebar.checkbox('Show raw Data', False):
            st.write("Cases of different Countries on ", dt)
            st.write(formatted_data)

    if(option == 'Future Prediction'):
        dt = '2020-07-30'
        coun_dt = cov_data[pd.to_datetime(
            cov_data['Date']) == pd.to_datetime(dt)]

        coun_dt.sort_values(
            by='Confirmed', ascending=False, inplace=True)

        list_countries = coun_dt['Country/Region'].to_list()
        list_countries.insert(0, 'World')
        list_countries = list(dict.fromkeys(list_countries))
        st.markdown("## Select a Country")
        selected_country = st.selectbox("", list_countries)
        type = st.sidebar.selectbox("", ['Confirmed', 'Recovered', 'Deaths'])
        period = st.sidebar.number_input(
            "Length of Period for Forecast (in Days)", 30, 365, step=1)
        if(selected_country == 'World'):
            world_aggregate_prophet = world_aggregate[['Date', type]]
            world_aggregate_prophet = world_aggregate_prophet.rename(
                columns={'Date': 'ds', type: 'y'})
            m = Prophet()
            m.fit(world_aggregate_prophet)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)
            fig = plot_plotly(m, forecast, xlabel='Date', ylabel=type, figsize=(
                800, 600))  # This returns a plotly Figure
            st.write(fig)

        else:
            selected_country_data = cov_data[cov_data['Country/Region']
                                             == selected_country]
            selected_country_data_prophet = selected_country_data[[
                'Date', type]]
            selected_country_data_prophet = selected_country_data_prophet.rename(
                columns={'Date': 'ds', type: 'y'})
            m = Prophet()
            m.fit(selected_country_data_prophet)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)
            fig = plot_plotly(m, forecast, xlabel='Date', ylabel=type, figsize=(
                800, 600))  # This returns a plotly Figure
            st.write(fig)


if __name__ == '__main__':
    main()
