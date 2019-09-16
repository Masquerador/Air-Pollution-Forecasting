"""
   ---------------------------------------------------------------------------------------------------------
    PYTHON PROJECT 2A
                      Air Pollution Forecasting using LSTM in KERAS
                                                                                         submitted by,
                                                                                         Justin Joseph.
                                                                                         18201354
   ---------------------------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------
    Import the necessary packages.
   ---------------------------------------------------------------------------------------------------------
"""
import warnings
import numpy as np
import math as mt
import datetime as dt
import pandas as pd
import sklearn.preprocessing as skp
import sklearn.metrics as skm
import keras.layers as kl
import keras.models as km
import bokeh.plotting as bp

warnings.filterwarnings("ignore")
"""
   ---------------------------------------------------------------------------------------------------------
    Function definitions.
   ---------------------------------------------------------------------------------------------------------
"""
def date_reformat(x): #Function to merge the individual date entries into a single column
    return dt.datetime.strptime(x,'%Y %m %d %H')

def timeseries_to_supervised(d,input=1,output=1): #Function to convert timeseries to supervised structure
    vars=1 if type(d) is list else d.shape[1]
    df=pd.DataFrame(d)
    cols,names=list(),list()
    #
    for i in range(input,0,-1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(vars)]
    #
    for i in range(0,output):
        cols.append(df.shift(-i))
        if i==0:
            names += [('var%d(t)' % (j+1)) for j in range(vars)]
        else:
            names += [('var%d(t+%d)' % (j+1,i)) for j in range(vars)]
    aggregate=pd.concat(cols,axis=1)
    aggregate.columns=names
    return aggregate

def drop_na(d): #Function to remove NA values
    d.dropna(inplace=True)
    return d

def create_lstm_model(data): #Function to create the LSTM model
    a = km.Sequential()
    a.add(kl.LSTM(100, input_shape=(data.shape[1], data.shape[2])))
    a.add(kl.Dense(1))
    a.compile(loss='mae', optimizer='adam')
    return a

def invert_scaling(a,b): #Function to perform inverse scaling
    c = np.concatenate((b, a[:, 1:]), axis=1)
    c = scale.inverse_transform(c)
    c = c[:, 0]
    return c
"""
   ---------------------------------------------------------------------------------------------------------
    Main Paragraph.
   ---------------------------------------------------------------------------------------------------------
"""
# Load the data with reformatted date and date column as the index
data=pd.read_csv('C:/Users/HP/Downloads/Study/Python/Assignment/Project 2/raw.csv',parse_dates=[['year','month','day',
                  'hour']],index_col=0,date_parser=date_reformat)
data.drop('No',axis=1,inplace=True) #Drop the column "No" as it has no significance in prediction
data.columns=['Pollution','Dew','Temperature','Pressure','Wind Direction','Wind Speed','Snow','Rain'] #Rename the columns
data.index.name='Date' #Rename the index
data=data[24:] #Drop the first 24 hours of data as it has NA values
data['Pollution'].fillna(0,inplace=True) #Replace all the other NA values with 0
val=data.values
"""
   ---------------------------------------------------------------------------------------------------------
    Bokeh plots to show the changes in the attributes with time.
   ---------------------------------------------------------------------------------------------------------
"""
bp.output_file("Pollution.html")
p=bp.figure(title='Pollution in Beijing during 2010-2014',x_axis_label='Year',y_axis_label='Pollution',
            x_axis_type='datetime',plot_width=1100)
p.line(data.index,val[:,0],legend=data.columns[0],line_width=2)
bp.show(p)

bp.output_file("Dew.html")
p=bp.figure(title='Dew in Beijing during 2010-2014',x_axis_label='Year',y_axis_label='Dew',
            x_axis_type='datetime',plot_width=1100)
p.line(data.index,val[:,1],legend=data.columns[1],line_width=2)
bp.show(p)

bp.output_file("Temperature.html")
p=bp.figure(title='Temperature in Beijing during 2010-2014',x_axis_label='Year',y_axis_label='Temperature',
            x_axis_type='datetime',plot_width=1100)
p.line(data.index,val[:,2],legend=data.columns[2],line_width=2)
bp.show(p)

bp.output_file("Pressure.html")
p=bp.figure(title='Pressure in Beijing during 2010-2014',x_axis_label='Year',y_axis_label='Pressure',
            x_axis_type='datetime',plot_width=1100)
p.line(data.index,val[:,3],legend=data.columns[3],line_width=2)
bp.show(p)

bp.output_file("Wind Speed.html")
p=bp.figure(title='Wind Speed in Beijing during 2010-2014',x_axis_label='Year',y_axis_label='Wind Speed',
            x_axis_type='datetime',plot_width=1100)
p.line(data.index,val[:,5],legend=data.columns[5],line_width=2)
bp.show(p)

bp.output_file("Snow.html")
p=bp.figure(title='Snow in Beijing during 2010-2014',x_axis_label='Year',y_axis_label='Snow',
            x_axis_type='datetime',plot_width=1100)
p.line(data.index,val[:,6],legend=data.columns[6],line_width=2)
bp.show(p)

bp.output_file("Rain.html")
p=bp.figure(title='Rain in Beijing during 2010-2014',x_axis_label='Year',y_axis_label='Rain',
            x_axis_type='datetime',plot_width=1100)
p.line(data.index,val[:,7],legend=data.columns[7],line_width=2)
bp.show(p)
"""
   ---------------------------------------------------------------------------------------------------------
    Prepare data for modelling.
   ---------------------------------------------------------------------------------------------------------
"""
# Convert categorical data (Wind Speed) into numbers
encode=skp.LabelEncoder()
val[:,4]=encode.fit_transform(val[:,4])

# Makes all data as float type
val=val.astype('float32')

# Normalize the data
scale=skp.MinMaxScaler(feature_range=(0,1))
scaled_val=scale.fit_transform(val)

# Reformat to supervised learning
reformatted=timeseries_to_supervised(scaled_val,1,1)

# Drop any NA values present after the shift operation is done
reformatted=drop_na(reformatted)

# Drop columns that aren't used to predict
reformatted.drop(reformatted.columns[[9,10,11,12,13,14,15]],axis=1,inplace=True)

# Split dataset into train and test datasets
val_reformat=reformatted.values
train_hours=(365*24)*4 # Train with Four years of data
train=val_reformat[:train_hours, :]
test=val_reformat[train_hours:, :]

# Split into input and output
train_X,train_Y=train[:, :-1],train[:, -1]
test_X,test_Y=test[:, :-1],test[:, -1]

# Reshape into a 3D array suitable for the LSTM model
train_X=train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X=test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
print(train_X.shape,train_Y.shape,test_X.shape,test_Y.shape)
"""
   ---------------------------------------------------------------------------------------------------------
    Design a neural network using LSTM .
   ---------------------------------------------------------------------------------------------------------
"""
model=create_lstm_model(train_X)

# Fit network
history=model.fit(train_X,train_Y,epochs=50,batch_size=72,validation_data=(test_X,test_Y),verbose=2,shuffle=False)
"""
   ---------------------------------------------------------------------------------------------------------
    Bokeh plot.
   ---------------------------------------------------------------------------------------------------------
"""
bp.output_file("Loss.html")
p=bp.figure(title='Loss',x_axis_label='Epoch',y_axis_label='Loss Value')
p.line(list(range(len(history.history['loss']))),history.history['loss'],legend='Train',line_color='firebrick',line_width=2)
p.line(list(range(len(history.history['val_loss']))),history.history['val_loss'],legend='Test',line_color='navy',line_width=2)
bp.show(p)
"""
   ---------------------------------------------------------------------------------------------------------
    Predict using the model generated above.
   ---------------------------------------------------------------------------------------------------------
"""
y_hat=model.predict(test_X)
test_X=test_X.reshape((test_X.shape[0],test_X.shape[2]))
test_Y=test_Y.reshape((len(test_Y),1))

# Invert scaling for actual
inv_Y=invert_scaling(test_X,test_Y)

# Invert scaling for forecast
inv_Y_hat=invert_scaling(test_X,y_hat)

# Calculate RMSE
RMSE=mt.sqrt(skm.mean_squared_error(inv_Y,inv_Y_hat))
print('The RMSE value is : ',RMSE)