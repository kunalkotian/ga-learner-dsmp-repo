# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(path)
data.hist(column='Rating')
data=data[data['Rating']<=5]
data.hist(column='Rating')
data



#Code starts here


#Code ends here


# --------------
# code starts here
import pandas as pd
total_null=data.isnull().sum()
percent_null=(total_null/data.isnull().count())
missing_data=pd.concat([total_null,percent_null],keys=['Total','Percent'],axis=1)
missing_data
data=data.dropna()
total_null_1=data.isnull().sum()
percent_null_1=total_null_1/data.isnull().count()
missing_data_1=pd.concat([total_null_1,percent_null_1],keys=['Total','Percent'],axis=1)
missing_data_1



# code ends here


# --------------

#Code starts here
import seaborn as sns
import matplotlib.pyplot as plt





a=sns.catplot(x="Category",y="Rating",data=data, kind="box",height = 10)
a.set_xticklabels(rotation=90)
plt.title('Rating vs Category [BoxPlot]')


#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
data['Installs'].value_counts()
data['Installs']=data['Installs'].str.replace(',', '')
data['Installs']=data['Installs'].str.replace('+', '')
data['Installs']= data['Installs'].astype(int) 
le=LabelEncoder()
data['Installs']= le.fit_transform(data['Installs']) 
a=sns.regplot(x="Installs",y="Rating",data=data)
plt.title('Rating vs Installs [RegPlot]')



# --------------
#Code starts here
data['Price'].value_counts()


data['Price']=data['Price'].str.replace('$', '')
data['Price']=data['Price'].astype(float)

ax=sns.regplot(x="Price",y="Rating",data=data)
plt.title('Rating vs Price  [RegPlot]')


#Code ends here


# --------------

#Code starts here
len(data['Genres'].unique())

data['Genres'] = data['Genres'].str.split(';').str[0]

#Grouping Genres and Rating
gr_mean=data[['Genres', 'Rating']].groupby(['Genres'], as_index=False).mean()

print(gr_mean.describe())

#Sorting the grouped dataframe by Rating
gr_mean=gr_mean.sort_values('Rating')

print(gr_mean.head(1))

print(gr_mean.tail(1))

#Code ends here


# --------------

#Code starts here
data['Last Updated'] = pd.to_datetime(data['Last Updated'])
max_date = data['Last Updated'].max()

data['Last Updated Days'] = max_date - data['Last Updated']
data['Last Updated Days'] = data['Last Updated Days'].dt.days
sns.regplot(x="Last Updated Days",y="Rating",data=data)
plt.title("Rating vs Category [BoxPlot]")
plt.show()



#Code ends here


