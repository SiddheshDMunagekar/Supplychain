from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import filterwarnings
from PIL import Image
filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from statsmodels.tsa.statespace.sarimax import SARIMAX
import sklearn.metrics  as metrics
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
st.set_option('deprecation.showPyplotGlobalUse', False)
url = 'https://drive.google.com/file/d/157aaIX8w9-tRGPF1qi00F0uXn0czFc9P/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]


df= pd.read_excel(path,parse_dates=True)
df.isnull().sum()
df.drop('Time',axis=1)
df=df.groupby('Date').mean()
df=df.rename(columns={'CO(GT)':'Carbon_monoxide','PT08.S1(CO)':'Tin_oxide','NMHC(GT)':'Non_methane_hydrocarbons','C6H6(GT)':'Benzene',
                   'PT08.S2(NMHC)':'titania','NOx(GT)':'Nitric_oxide','PT08.S3(NOx)':'tungsten_oxide_NO','NO2(GT)':'Nitrogen_dioxide',
                  'PT08.S4(NO2)':'tungsten_oxide_NO2','PT08.S5(O3)':'indium_oxide' }, inplace=False)
#st.write(df)




def run_app():
    image=Image.open('Air_pol.JPG')

    st.image(image,use_column_width=True)
    no2=Image.open('no2.JPG')
    st.sidebar.image(no2,use_column_width=True)


    df2 = df.copy()
    df2 = df2['Nitrogen_dioxide']
    train = df2[0:-30]
    test = df2[-30:]

    add_selectbox = st.sidebar.selectbox("Select Forecasting Model", ("Simple Moving Average", "LSTM","Triple Exponential Smoothing","Seasonal ARIMA",
                                                               "Gradient Boosting Regressor","ML Model Comparison Table"))
    st.sidebar.info('This application focuses on forecasting Nitrogen dioxide concentration in air using multiple forecasting technique')
    st.sidebar.info('Supply chain Analytics Final Project DAB 400 Group 10')


    if add_selectbox == 'Simple Moving Average':
        df1 = df.Nitrogen_dioxide.copy()
        df1 = pd.DataFrame(df1)
        df1['SMA_20'] = df1.Nitrogen_dioxide.rolling(20, min_periods=1).mean()
        df1['SMA_10'] = df1.Nitrogen_dioxide.rolling(10, min_periods=1).mean()
        df1['SMA_3'] = df1.Nitrogen_dioxide.rolling(3, min_periods=1).mean()
        fig = plt.figure()

        df1.plot(figsize=(25, 15))
        plt.xlabel('Date',fontsize=20)
        plt.ylabel('Nitrogen dioxide',fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title("Simple Moving  Average for 20, 10 and 3 days", fontsize=30)
        plt.legend(labels=['Temperature','20-days SMA','10-days SMA','3-days SMA'],fontsize=22)
        plt.grid()
        plt.show()
        st.pyplot(use_column_width=True)
        mae = metrics.mean_absolute_error(df['Nitrogen_dioxide'], df1['SMA_20'])
        st.write("MAE for 20 days is {:,.2f}".format(mae))
        mae = metrics.mean_absolute_error(df['Nitrogen_dioxide'], df1['SMA_10'])
        st.write("MAE for 10 days is {:,.2f}".format(mae))
        mae = metrics.mean_absolute_error(df['Nitrogen_dioxide'], df1['SMA_3'])
        st.write("MAE for 3 days is {:,.2f}".format(mae))



    if add_selectbox == 'Triple Exponential Smoothing':

        train = pd.DataFrame(train)
        test = pd.DataFrame(test)
        pred = test.copy()
        fit1 = ExponentialSmoothing(np.asarray(train['Nitrogen_dioxide']), trend='add', seasonal_periods=7,
                                    seasonal='add').fit()

        pred['Holt_Winter'] = fit1.forecast(len(test))
        # Calculate KPI's
        mae = metrics.mean_absolute_error(test.Nitrogen_dioxide, pred.Holt_Winter)


        # Plot
        plt.figure(figsize=(16, 8))
        plt.plot(train['Nitrogen_dioxide'], label='Train')
        plt.plot(test['Nitrogen_dioxide'], label='Test')
        plt.plot(pred['Holt_Winter'], label='Holt_Winter (MAE={:.2f})'.format(mae))
        plt.title("Triple Exponential smoothing",fontsize=30)
        plt.xlabel('Date',fontsize=20)
        plt.ylabel('Nitrogen dioxide',fontsize=20)
        plt.legend(fontsize=19)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.show()
        st.pyplot(use_column_width=True)
        st.write("MAE for 30 days is {:,.2f}".format(mae))


    ##Seasonal_Arima
    if add_selectbox=='Seasonal ARIMA':

        df3 = df.copy()
        #train = df3[0:-30]
        test = df3[-30:]

        model = SARIMAX(df3['Nitrogen_dioxide'], order=(0, 1, 0), seasonal_order=(2, 1, 0, 30),
                        enforce_stationarity=False,
                        enforce_invertibility=False, dynamic=True)
        results = model.fit()


        df3['predicted_test'] = results.predict(start=360, end=390, dynamic=True)

        seasonal_forecast = pd.DataFrame(results.forecast(len(test)))
        seasonal_forecast = seasonal_forecast.rename({0: 'Seasonal forecast for 30 periods'}, axis=1)

        plt.figure(figsize=(16, 8))
        seasonal_forecast.plot(figsize=(25, 10), color='green')
        df3['Nitrogen_dioxide'].plot(figsize=(20, 10))
        df3['predicted_test'].plot(figsize=(20, 10))
        plt.legend(fontsize=19)
        plt.ylabel("Nitrogen_dioxide",fontsize=20)
        plt.xlabel('Date', fontsize=20)
        plt.title("Seasonal Arima",fontsize=30)
        plt.legend(fontsize=19)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid()
        plt.show()
        st.pyplot(use_column_width=True)

        # Calculate KPI
        mae = metrics.mean_absolute_error(df3.Nitrogen_dioxide[360:], df3.predicted_test[360:])

        st.write("MAE of Seasonal Arima is  {:.2f}".format(mae))

    if add_selectbox=='ML Model Comparison Table':

        acc_table = {'Model': ['Linear Regression', 'Decision Tree', 'Random_forest', 'Gradient_Boosting'],
                     'Train_score': [0.59, 0.70, 0.91, 0.83],
                     'Test_score': [0.49, 0.46, 0.40, 0.50],
                     'MAE_train': [4265.36, 3713.86, 2220.76, 2958.29], 'MAE_test': [3053.96, 3116.57, 3161.29, 2726.36]
                     }
        acc_table = pd.DataFrame(acc_table)
        acc_table=acc_table.sort_values(by='Test_score', ascending=False).reset_index(drop=True)
        st.table(acc_table)



    #Gradient Boosting
    if add_selectbox=='Gradient Boosting Regressor':

        df55 = df.Nitrogen_dioxide.copy()
        df55 = pd.DataFrame(df55)
        dfML = pd.DataFrame()
        for i in range(7, 0, -1):
            dfML[['t-' + str(i)]] = df55.shift(i)

        dfML['t'] = df55.values
        df_ML = dfML[7:]
        # Split Data into dependent(target) and independent(features) variables

        df_ML22 = df_ML.values
        # Lagged variables (features) and original time series data (target)
        X2 = df_ML22[:, 0:-1]  # slice all rows and start with column 0 and go up to but not including the last column
        y2 = df_ML22[:, -1]  # slice all rows and last column, essentially separating out 't' column

        traintarget_size = int(len(y2) * 0.70)
        train_target, test_target = y2[:traintarget_size], y2[traintarget_size:len(y2)]
        trainfeature_size = int(len(X2) * 0.70)
        train_feature, test_feature = X2[:trainfeature_size], X2[trainfeature_size:len(X2)]



        gbr = GradientBoostingRegressor(max_features=3,
                                        max_depth=2,
                                        learning_rate=0.1,
                                        n_estimators=100,
                                        subsample=0.8,
                                        random_state=50)

        gbr.fit(train_feature, train_target)

        gbr_train_70_30 = gbr.score(train_feature, train_target)
        gbr_test_70_30 = gbr.score(test_feature, test_target)

        plot_test_pred = gbr.predict(test_feature)
        plot_test_pred = pd.DataFrame(plot_test_pred)
        plot_test_pred = plot_test_pred.rename({0: 'Predicted_test'}, axis=1)

        plot_test_target = pd.DataFrame(test_target)
        plot_test_target = plot_test_target.rename({0: 'Actual_test'}, axis=1)
        gbr_test_plot = pd.concat([plot_test_target, plot_test_pred], axis=1)

        gbr_test_plot.plot(title='Gradient boosting Actual vs Predicted test of last 116 days')
        plt.grid()
        st.pyplot(use_column_width=True)
        st.write("Gradient boosting training score {:.2f}".format(round(gbr_train_70_30, 2)))
        st.write("Gradient boosting test score {:.2f}".format(round(gbr_test_70_30, 2)))

        if st.checkbox("Visualize for last 10 days"):
            gbr_test_plot[106:].plot(title=' GBR Plot of last 10 days')
            st.pyplot(use_column_width=True)






    if add_selectbox=='LSTM':

        data = df.copy()
        data = data.iloc[:, 7].values
        data = data.reshape(-1, 1)
        data = data.astype('float32')

        # Scaling the data
        scalar = MinMaxScaler()
        data = scalar.fit_transform(data)

        train_lstm = data[:-30, :]
        test_lstm = data[-30:, :]

        # Building the 2D array for supervised learning
        def create_dataset(sequence, time_step):
            dataX = []
            dataY = []
            for i in range(len(sequence) - time_step - 1):
                a = sequence[i:(i + time_step), 0]
                dataX.append(a)
                dataY.append(sequence[i + time_step, 0])
            return np.array(dataX), np.array(dataY)

        time_step = 1
        # Apply the 2D array function to train and test datasets
        train_X, train_Y = create_dataset(train_lstm, time_step)
        test_X, test_Y = create_dataset(test_lstm, time_step)

        train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
        test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))


        # Build the LSTM Model
        model = Sequential()
        # Adding the input layer and LSTM layer
        model.add(LSTM(50, activation='relu', input_shape=(1, time_step), return_sequences=True))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50))
        model.add(Dropout(0.15))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(train_X, train_Y, batch_size=4, epochs=50, verbose=2)

        # Make predictions


        train_predict = model.predict(train_X)
        test_predict = model.predict(test_X)
        # inverting predictions
        train_predict = scalar.inverse_transform(train_predict)
        train_Y = scalar.inverse_transform([train_Y])
        test_predict = scalar.inverse_transform(test_predict)
        test_Y = scalar.inverse_transform([test_Y])
        # calculate root mean squared error
        train_score = mean_absolute_error(train_Y[0], train_predict[:, 0])

        test_score = mean_absolute_error(test_Y[0], test_predict[:, 0])


        # LSTM plot
        train_plot = np.empty_like(data)  # create an array with the same shape as provided
        train_plot[:, :] = np.nan
        train_plot[time_step:len(train_predict) + time_step, :] = train_predict
        # shifting test predictions for plotting
        test_plot = np.empty_like(data)
        test_plot[:, :] = np.nan
        test_plot[len(train_predict) + (time_step * 2) + 1:len(data) - 1, :] = test_predict
        # plot baseline and predictions
        plt.figure(figsize=(16, 8))
        
        plt.plot(train_plot)
        plt.plot(test_plot)
        plt.plot(scalar.inverse_transform(data))
        plt.title("Long Short Term Memory Network with train ,test and forecast", fontsize=20)
        plt.ylabel("Nitrogen_dioxide", fontsize=20)
        plt.legend(labels=['Original plot','Test set','LSTM forecast'],fontsize=19)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.grid()


        plt.show()
        st.pyplot(use_column_width=True)

        st.write('Train Score: %.3f MAE' % (train_score))
        st.write('Test Score: %.3f MAE' % (test_score))

        if st.checkbox('Visualize forecasted chart for 10 future days'):
            test_predict = scalar.fit_transform(test_predict)
            time_step = 10
            x_input = test_predict[(len(test_predict) - time_step):].reshape(1, -1)
            # Converting it to list
            temp_input = list(x_input)
            # Arranging list vertically
            temp_input = temp_input[0].tolist()

            # demonstrate prediction for next 10 days

            lst_output = []
            future_day = 10
            n_steps = 10
            i = 0
            # Forcast next 10 days output
            while (i < future_day):

                if (len(temp_input) > 10):

                    x_input = np.array(temp_input[1:])
                    print("{} day input {}".format(i, x_input))
                    x_input = x_input.reshape(1, -1)
                    # Converting to 3d array for lstm
                    x_input = x_input.reshape(1, n_steps, 1)
                    # print(x_input)
                    ypred = model.predict(x_input, verbose=0)
                    print("{} day predicted output {}".format(i, ypred))
                    # adding predicted output  to temp_input list
                    temp_input.extend(ypred[0].tolist())
                    temp_input = temp_input[1:]

                    # print(temp_input)
                    lst_output.extend(ypred.tolist())
                    i = i + 1
                else:
                    x_input = x_input.reshape((n_steps, 1, 1))
                    ypred = model.predict(x_input, verbose=0)
                    print("Predicted y of 0 day", ypred[0])
                    # Addding ypred value in temp_input(previous input)
                    temp_input.extend(ypred[0].tolist())
                    print(len(temp_input))
                    lst_output.extend(ypred.tolist())
                    i = i + 1
                # print(lst_output)

            previous_days1 = np.arange(len(data) - n_steps, len(data))
            predicted_future1 = np.arange(len(data), len(data) + future_day)
            lst_output = lst_output[:future_day]
            outputlist = data.tolist()
            outputlist.extend(lst_output)
            #data[len(data) - n_steps:]

            plt.plot(np.append(previous_days1, predicted_future1),
                     scalar.inverse_transform(outputlist[len(data) - n_steps:]))
            plt.plot(predicted_future1, scalar.inverse_transform(lst_output))
            plt.title("Forecast for 10 future days",fontsize=20)
            plt.legend(fontsize=19)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=8)
            plt.ylabel("Nitrogen dioxide")
            plt.show()
            st.pyplot(use_column_width=True)



if __name__=='__main__':
    run_app()
