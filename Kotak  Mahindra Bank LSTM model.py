#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from datetime import datetime
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Importing Training Set
dataset_train = pd.read_csv("D:\Abou kotak bank dataset\Abou Nasar. KOTAKBANK.NS.csv")
dataset_train


# In[3]:


# Select features (columns) to be involved intro training and predictions
cols = list(dataset_train)[1:6]
cols


# In[4]:


# Extract dates (will be used in visualization)
datelist_train = list(dataset_train['Date'])
datelist_train = [dt.datetime.strptime(date, '%Y-%m-%d').date() for date in datelist_train]


# In[5]:


print('Training set shape == {}'.format(dataset_train.shape))
print('All timestamps == {}'.format(len(datelist_train)))
print('Featured selected: {}'.format(cols))


# In[6]:


df = pd.DataFrame(dataset_train)


# In[7]:


df


# In[8]:


#df.set_index('Date', inplace=True)
df.plot()


# In[9]:


#Step #2. Data pre-processing
#Removing all commas and convert data to matrix shape format.

dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0, len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(',', '')

dataset_train = dataset_train.astype(float)


# In[10]:


dataset_train.head(5)


# In[11]:


# Using multiple features (predictors)
training_set = dataset_train.values

print('Shape of training set == {}.'.format(training_set.shape))
training_set


# In[12]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[13]:


sc = StandardScaler()
training_set_scaled = sc.fit_transform(training_set)


# In[14]:


training_set_scaled


# In[15]:


sc_predict = StandardScaler()
sc_predict.fit_transform(training_set[:, 0:1])


# In[16]:


# Creating a data structure with 90 timestamps and 1 output
X_train = []
y_train = []


# In[17]:


n_future = 60   # Number of days we want top predict into the future
n_past = 90     # Number of past days we want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future +1):
    X_train.append(training_set_scaled[i - n_past:i, 0:dataset_train.shape[1] - 1])
    y_train.append(training_set_scaled[i + n_future - 1:i + n_future, 0])

X_train, y_train = np.array(X_train), np.array(y_train)


# In[18]:


print('X_train shape == {}.'.format(X_train.shape))
print('y_train shape == {}.'.format(y_train.shape))


# In[19]:


#PART 2. Create a model. Training
#Step #3. Building the LSTM based Neural Network


# In[20]:


# Import Libraries and packages from Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow.keras.optimizers import Adam


# In[21]:


# Initializing the Neural Network based on LSTM
model = Sequential()


# In[22]:


# Adding 1st LSTM layer
model.add(LSTM(units=64, return_sequences=True, input_shape=(n_past, dataset_train.shape[1]-1)))


# In[23]:


# Adding 2nd LSTM layer
model.add(LSTM(units=10, return_sequences=False))


# In[24]:


# Adding Dropout
model.add(Dropout(0.25))


# In[25]:


# Output layer
model.add(Dense(units=1, activation='linear'))


# In[76]:


# Compiling the Neural Network
model.compile(optimizer = Adam(learning_rate=0.01), loss='mean_squared_error')
model.fit(X_train,y_train,epochs=10)


# In[86]:


model.compile(optimizer = 'sgd', loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy'])
model.fit(X_train,y_train,epochs=5,batch_size=10 )


# In[28]:


model.summary()


# In[29]:


model.save('keras_model.h5')


# In[30]:


#The last date for our training set is 30-Dec-2016.
#We will perform predictions for the next 20 days, since 2017-01-01 to 2017-01-20.


# In[31]:


#PART 3. Make future predictions


# In[32]:


# Generate list of sequence of days for predictions
datelist_future = pd.date_range(datelist_train[-1], periods=n_future, freq='1d').tolist()

'''
Remeber, we have datelist_train from begining.
'''


# In[33]:


# Convert Pandas Timestamp to Datetime object (for transformation) --> FUTURE
datelist_future_ = []
for this_timestamp in datelist_future:
    datelist_future_.append(this_timestamp.date())


# In[34]:


#Step #5. Make predictions for future dates


# In[35]:


# Perform predictions
predictions_future = model.predict(X_train[-n_future:])

predictions_train = model.predict(X_train[n_past:])


# In[36]:


predictions_future


# In[37]:


predictions_train


# In[38]:


# Inverse the predictions to original measurements


# In[39]:


# ---> Special function: convert <datetime.date> to <Timestamp>
def datetime_to_timestamp(x):
    '''
        x : a given datetime value (datetime.date)
    '''
    return datetime.strptime(x.strftime('%Y%m%d'), '%Y%m%d')


# In[40]:


y_pred_future = sc_predict.inverse_transform(predictions_future)
y_pred_train = sc_predict.inverse_transform(predictions_train)


# In[41]:


PREDICTIONS_FUTURE = pd.DataFrame(y_pred_future, columns=['Open']).set_index(pd.Series(datelist_future))
PREDICTION_TRAIN = pd.DataFrame(y_pred_train, columns=['Open']).set_index(pd.Series(datelist_train[2 * n_past + n_future -1:]))


# In[42]:


PREDICTIONS_FUTURE.tail(13)


# In[43]:


PREDICTION_TRAIN.tail(13)


# In[92]:


df.tail(13)


# In[44]:


# Convert <datetime.date> to <Timestamp> for PREDCITION_TRAIN
PREDICTION_TRAIN.index = PREDICTION_TRAIN.index.to_series().apply(datetime_to_timestamp)


# In[45]:


PREDICTION_TRAIN.index


# In[46]:


# Set plot size 
from pylab import rcParams
rcParams['figure.figsize'] = 14, 5
rcParams['figure.figsize']


# In[47]:


# Plot parameters
START_DATE_FOR_PLOTTING = '2018-01-01'


# In[48]:


plt.plot(PREDICTIONS_FUTURE.index, PREDICTIONS_FUTURE['Open'], color='r', label='Predicted Stock Price')
plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')


# In[49]:


plt.plot(PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:].index, PREDICTION_TRAIN.loc[START_DATE_FOR_PLOTTING:]['Open'], color='orange', label='Training predictions')


# In[50]:


#plt.plot(dataset_train.loc[START_DATE_FOR_PLOTTING:].index, dataset_train.loc[START_DATE_FOR_PLOTTING:]['Open'], color='b', label='Actual Stock Price')

plt.axvline(x = min(PREDICTIONS_FUTURE.index), color='green', linewidth=2, linestyle='--')


# In[51]:


plt.grid(which='major', color='red', alpha=0.8)


# In[52]:


plt.legend(shadow=True)
plt.title('Predcitions and Acutal Stock Prices', family='Arial', fontsize=12)
plt.xlabel('Timeline', family='Arial', fontsize=10)
plt.ylabel('Stock Price Value', family='Arial', fontsize=10)
plt.xticks(rotation=30, fontsize=8)
plt.show()


# In[53]:


# Parse training set timestamp for better visualization
dataset_train = pd.DataFrame(dataset_train, columns=cols)
dataset_train.index = datelist_train
dataset_train.index = pd.to_datetime(dataset_train.index)


# In[54]:


dataset_train
dataset_train.index


# In[55]:


# tkinter GUI part.


# In[56]:


import tkinter as tk
from tkinter import messagebox
from keras.models import load_model
import numpy as np


# In[57]:


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# In[58]:


root=tk.Tk()


# In[59]:


canvas1=tk.Canvas(root,width=500,height=500)
canvas1.pack()


# In[60]:


# New_Open label input box

label1=tk.Label(root,text='Open:')
canvas1.create_window(100,100,window=label1)
label1.config(font=("Times",18))


# In[61]:


entry1=tk.Entry(root) # create first entry box.
canvas1.create_window(380,100,window=entry1)
entry1.config(font=("Times",18))


# In[62]:


# New_High label input box

label2=tk.Label(root,text='High:')
canvas1.create_window(150,140,window=label2)
label2.config(font=("Times",18))


# In[63]:


entry2=tk.Entry(root) # create second entry box.
canvas1.create_window(420,130,window=entry2)
entry2.config(font=("Times",18))


# In[64]:


# New_Low label input box

label3=tk.Label(root,text='Low:')
canvas1.create_window(195,170,window=label3)
label3.config(font=("Times",18))


# In[65]:


entry3=tk.Entry(root) # create Third entry box.
canvas1.create_window(440,160,window=entry3)
entry3.config(font=("Times",18))


# In[66]:


# New_Close label input box

label4=tk.Label(root,text='Close:')
canvas1.create_window(240,200,window=label4)
label4.config(font=("Times",18))


# In[67]:


entry4=tk.Entry(root) # create Fourth entry box.
canvas1.create_window(470,190,window=entry4)
entry4.config(font=("Times",18))


# In[68]:


#lstm model load
model=load_model('keras_model.h5')


# In[69]:


# Tkinter GUI class
class StockPricePredictionGUI:
    def _init_(self, root):
        self.root = root
        self.root.title("Stock Price Prediction")
        
        self.label = tk.Label(root, text="Enter the previous stock prices:")
        self.label.pack()
        
        self.entry = tk.Entry(root)
        self.entry.pack()
        
        self.button = tk.Button(root, text="Predict", command=self.predict_stock_price)
        self.button.pack()
        
    def predict_stock_price(self):
        # User dvara enter kiye gaye stock prices ko lein
        prices = self.entry.get()
        
        # Stock prices ko array mein convert karein
        prices_array = np.array([float(price) for price in prices.split(',')])
        
        # LSTM model ke liye data reshape karein
        prices_array = np.reshape(prices_array, (1, len(prices_array), 1))
        
        # Stock price predict karein
        predicted_price = model.predict(prices_array)
        
        # Predicted price ko GUI mein display karein
        messagebox.showinfo("Predicted Price", f"The predicted stock price is: {predicted_price[0][0]}")
app = StockPricePredictionGUI()


# In[70]:


# Tkinter GUI class
class StockPricePredictionGUI:
    def _init_(self, root):
        self.root = root
        self.root.title("Stock Price Prediction")
        
        # Global market adjusted factor options
        self.gma_factors = [0.9, 0.95, 1.0, 1.05, 1.1]
        
        self.label = tk.Label(root, text="Enter the previous stock prices:")
        self.label.pack()
        
        self.entry = tk.Entry(root)
        self.entry.pack()
        
        self.gma_label = tk.Label(root, text="Select the Global Market Adjusted Factor:")
        self.gma_label.pack()
        
        # Dropdown menu
        self.gma_var = tk.StringVar(root)
        self.gma_var.set(self.gma_factors[2])  # Default value
        self.gma_dropdown = tk.OptionMenu(root, self.gma_var, *self.gma_factors)
        self.gma_dropdown.pack()
        
        self.button = tk.Button(root, text="Predict", command=self.predict_stock_price)
        self.button.pack()
        
    def predict_stock_price(self):
        # User dvara enter kiye gaye stock prices ko lein
        prices = self.entry.get()
        
        # Stock prices ko array mein convert karein
        prices_array = np.array([float(price) for price in prices.split(',')])
        
        # LSTM model ke liye data reshape karein
        prices_array = np.reshape(prices_array, (1, len(prices_array), 1))
        
        # Global market adjusted factor ko lein
        gma_factor = float(self.gma_var.get())
        
        # Stock price predict karein
        predicted_price = model.predict(prices_array)
        
        # Global market adjusted factor ko apply karein
        predicted_price *= gma_factor
        
        # Predicted price ko GUI mein display karein
        messagebox.showinfo("Predicted Price", f"The predicted stock price is: {predicted_price[0][0]}")


app = StockPricePredictionGUI()


# In[71]:


# Tkinter GUI class
class StockPricePredictionGUI:
    def _init_(self, root):
        self.root = root
        self.root.title("Stock Price Prediction")
        
        # Global market sentiment options
        self.sentiments = ['Positive', 'Neutral', 'Negative']
        
        self.label = tk.Label(root, text="Open:")
        self.label.pack()
        
        self.entry = tk.Entry(root)
        self.entry.pack()
        
        self.sentiment_label = tk.Label(root, text="Select the Global Market Sentiment:")
        self.sentiment_label.pack()
        
        # Dropdown menu
        self.sentiment_var = tk.StringVar(root)
        self.sentiment_var.set(self.sentiments[1])  # Default value
        
        self.sentiment_dropdown = tk.OptionMenu(root, self.sentiment_var, *self.sentiments)
        self.sentiment_dropdown.pack()
        
        self.button = tk.Button(root, text="Predict", command=self.PREDICTIONS_FUTURE)
        self.button.pack()
    def predict_stock_price(self):
        # User dvara enter kiye gaye stock prices ko lein
        prices = self.entry.get()
        
        # Stock prices ko array mein convert karein
        prices_array = np.array([float(price) for price in prices.split(',')])
        
        # LSTM model ke liye data reshape karein
        prices_array = np.reshape(prices_array, (1, len(prices_array), 1))
        
        # Global market sentiment ko lein
        sentiment = self.sentiment_var.get()
        
        # Stock price predict karein
        PREDICTIONS_FUTURE = model.predict(prices_array)
        
        # Global market sentiment ke hisab se stock price ko adjust karein
        if sentiment == 'Positive':
            PREDICTIONS_FUTURE *= 1.1  # Positive sentiment multiplier
        elif sentiment == 'Neutral':
            PREDICTIONS_FUTURE *= 1.0  # Neutral sentiment multiplier
        else:
            PREDICTIONS_FUTURE *= 0.9  # Negative sentiment multiplier
        
        # Predicted price ko GUI mein display karein
        messagebox.showinfo("PREDICTIONS_FUTURE", f"The predicted stock price is: {PREDICTIONS_FUTURE[0][0]}")


app = StockPricePredictionGUI()


# In[72]:


# GUI ko run karein
root.mainloop()


# In[ ]:




