#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime as dt
from datetime import datetime
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model


# In[2]:


df=pd.read_csv("D:\Abou kotak bank dataset\Abou Nasar. KOTAKBANK.NS.csv")
df


# In[3]:


df.isnull().sum()


# In[4]:


df.corr()


# In[5]:


sns.heatmap(df.corr())


# In[6]:


df.head(5)


# In[7]:


df.tail(5)


# In[8]:


df.plot()


# In[9]:


sns.relplot(data=df,x='Date',y='Close',hue="Open")


# In[10]:


sns.relplot(data=df,x='High',y='Close',hue="Volume")


# In[11]:


sns.relplot(data=df,x='Low',y='Close',hue="Volume")


# In[12]:


sns.relplot(data=df,x='Date',y='Close',hue="Volume")


# In[13]:


sns.relplot(data=df,x='Adj Close',y='Close',hue="Volume")


# In[14]:


sns.relplot(data=df,x='Volume',y='Close',hue="Volume")


# In[15]:


x= df[['High','Open','Low','Volume']].values
y=df['Close'].values


# In[16]:


#split the data into 80% training and 20% testing.
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=5)


# In[17]:


x_train


# In[18]:


x_test


# In[19]:


y_train


# In[20]:


y_test


# In[21]:


reg=LinearRegression()
reg.fit(x_train,y_train)


# In[22]:


print('Intercept:\n',reg.intercept_)
print('Coefficients:\n',reg.coef_)


# In[23]:


accuracy=reg.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[24]:


from sklearn import tree


# In[25]:


tree_model= DecisionTreeRegressor()
tree_model.fit(x_train,y_train)


# In[26]:


y_pred=tree_model.predict(x_test)


# In[27]:


accuracy=tree_model.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[28]:


clf=linear_model.Lasso()
clf.fit(x_train,y_train)


# In[29]:


accuracy=clf.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[30]:


print('Intercept:\n',clf.intercept_)
print('Coefficients:\n',clf.coef_)


# In[31]:


y_pred=clf.predict(x_test)


# In[32]:


predicted1=clf.predict(x_test)
expected=y_test
print(predicted1)


# In[33]:


predicted1 = clf.predict(x_test)
result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted1':predicted1.flatten()})
result.head(30)


# In[34]:


import math
graph=result.head(30)
graph.plot(kind='bar')


# In[35]:


accuracy=clf.score(x_test,y_test)
print("Accuracy:",accuracy)


# In[36]:


predicted=reg.predict(x_test)
expected=y_test
print(predicted)


# In[37]:


predicted = reg.predict(x_test)
result = pd.DataFrame({'Actual':y_test.flatten(),'Predicted':predicted.flatten()})
result.head(11)


# In[38]:


import math
graph=result.head(30)
graph.plot(kind='bar')


# In[39]:


#visualzied the Open price data
plt.figure(figsize=(16,8))
plt.title('KOTAkBank')
plt.xlabel('Days')
plt.ylabel('Open Price (RS)')
plt.plot(df['Open'])
plt.show()


# In[40]:


df.Close


# In[41]:


#create a variable to predict 'x' days out into the future
future_days =0
df['Predicted']=df[['Close']].shift(future_days)
df.tail(30)


# In[42]:


future_days=0
df['Predicted']=df[['Close']].shift(future_days)
df.head(50)


# In[43]:


#create a feature data set(x) and convert it to a numpy array and remove the last 'x' rows/days
x=np.array(df.drop(['Predicted'],1))[:-future_days]
print(x)


# In[44]:


#create the target data set (y) and convert it to a numpy array and get all the target values except the last 'x' rows/days
y=np.array(df['Predicted'])[:-future_days]
print(y)


# In[45]:


#get the last 'x' rows of the feature data set
x_future = df.drop(['Predicted'],1)[:-future_days]
x_future=np.array(x_future)
x_future


# In[46]:


# tkinter GUI part.


# In[47]:


import tkinter as tk


# In[48]:


import matplotlib.pyplot as plt


# In[49]:


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# In[50]:


root=tk.Tk()


# In[51]:


canvas1=tk.Canvas(root,width=500,height=500)
canvas1.pack()


# In[52]:


# with sklearn


# In[53]:


Intercept_result=('Intercept:',reg.intercept_)
label_Intercept=tk.Label(root,text=Intercept_result,justify='center')
canvas1.create_window(280,230,window=label_Intercept)
label_Intercept.config(font=("Times",14))


# In[54]:


# with sklearn

Coefficients_result=('Coefficients:',reg.coef_)
label_Coefficients=tk.Label(root,text=Coefficients_result,justify='center')
canvas1.create_window(317,275,window=label_Coefficients)
label_Coefficients.config(font=("Times",14))


# In[55]:


# with sklearn

Intercept_result1=('Intercept:',clf.intercept_)
label_Intercept1=tk.Label(root,text=Intercept_result1,justify='center')
canvas1.create_window(280,250,window=label_Intercept1)
label_Intercept1.config(font=("Times",14))


# In[56]:


# with sklearn

Coefficients_result1=('Coefficients:',clf.coef_)
label_Coefficients1=tk.Label(root,text=Coefficients_result1,justify='center')
canvas1.create_window(342,295,window=label_Coefficients1)
label_Coefficients1.config(font=("Times",14))


# In[57]:


# New_Open label input box

label1=tk.Label(root,text='Open:')
canvas1.create_window(100,100,window=label1)
label1.config(font=("Times",18))


# In[58]:


entry1=tk.Entry(root) # create first entry box.
canvas1.create_window(380,100,window=entry1)
entry1.config(font=("Times",18))


# In[59]:


# New_High label input box

label2=tk.Label(root,text='High:')
canvas1.create_window(150,140,window=label2)
label2.config(font=("Times",18))


# In[60]:


entry2=tk.Entry(root) # create second entry box.
canvas1.create_window(420,130,window=entry2)
entry2.config(font=("Times",18))


# In[61]:


# New_Low label input box

label3=tk.Label(root,text='Low:')
canvas1.create_window(195,170,window=label3)
label3.config(font=("Times",18))


# In[62]:


entry3=tk.Entry(root) # create Third entry box.
canvas1.create_window(440,160,window=entry3)
entry3.config(font=("Times",18))


# In[63]:


# New_Close label input box

label4=tk.Label(root,text='Close:')
canvas1.create_window(240,200,window=label4)
label4.config(font=("Times",18))


# In[64]:


entry4=tk.Entry(root) # create Fourth entry box.
canvas1.create_window(470,190,window=entry4)
entry4.config(font=("Times",18))


# In[65]:


def values():
    global New_Open #our 1st input variable.
    New_Open=float(entry1.get())
    
    global New_High #our 2nd input variable.
    New_High=float(entry2.get())
    
    global New_Low #our 3rd input variable.
    New_Low=float(entry3.get())
    
    global New_Close #our 4th input variable.
    New_Close=float(entry4.get())
    
    Prediction_result=('Predicted :',reg.predict([[New_Open,New_High,New_Low,New_Close]]))
    label_Prediction=tk.Label(root,text=Prediction_result,bg='Red')
    canvas1.create_window(290,340,window=label_Prediction)
    label_Prediction.config(font=("Times",18))
    
button1=tk.Button(root,text='Predict ',command=values,bg='yellow')  #button to call.
canvas1.create_window(70,340,window=button1)
button1.config(font=("Times",18))


# In[66]:


# plot 1st scatter

figure3=plt.figure(figsize=(3,2),dpi=100)
ax3=figure3.add_subplot(111)
ax3.scatter(df['Open'].astype(float),df['Close'].astype(float),color='r')
scatter3=FigureCanvasTkAgg(figure3,root)
scatter3.get_tk_widget().pack(side=tk.RIGHT,fill=tk.BOTH)
ax3.legend(['Close'])
ax3.set_xlabel('Open')
ax3.set_title('Open Vs. Close')


# In[67]:


# plot 3rd scatter

figure5=plt.figure(figsize=(3,2),dpi=100)
ax5=figure5.add_subplot(111)
ax5.scatter(df['High'].astype(float),df['Close'].astype(float),color='g')
scatter5=FigureCanvasTkAgg(figure5,root)
scatter5.get_tk_widget().pack(side=tk.RIGHT,fill=tk.BOTH)
ax5.legend(['Close'])
ax5.set_xlabel('High')
ax5.set_title('High Vs. Close')


# In[68]:


# plot 4th scatter

figure6=plt.figure(figsize=(3,2),dpi=100)
ax6=figure6.add_subplot(111)
ax6.scatter(df['Low'].astype(float),df['Close'].astype(float),color='r')
scatter6=FigureCanvasTkAgg(figure6,root)
scatter6.get_tk_widget().pack(side=tk.RIGHT,fill=tk.BOTH)
ax6.legend(['Close'])
ax6.set_xlabel('Low')
ax6.set_title('Low Vs. Close')


# In[ ]:


root.mainloop()


# In[ ]:




