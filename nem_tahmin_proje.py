# -*- coding: utf-8 -*-


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('tenis_veriler.csv')
print(veriler)

#verilere baktığımızda numeric veriye çevirmemiz gereken stunlar var (windy,play)

#outlook kategorik--> numeric
hava=veriler.iloc[:,0:1].values
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
hava[:,0] = le.fit_transform(veriler.iloc[:,0])#encoding işlemi
ohe = preprocessing.OneHotEncoder()
hava = ohe.fit_transform(hava).toarray()#3 e ayırma


sicaklik=veriler.iloc[:,1:2].values


nem=veriler.iloc[:,2:3].values


#encoder: Kategorik -> Numeric
ruzgar= veriler.iloc[:,3:4].values#ruzgarı ayırma
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ruzgar[:,-1] = le.fit_transform(veriler.iloc[:,3:4])#encoding işlemi
ohe = preprocessing.OneHotEncoder()
ruzgar= ohe.fit_transform(ruzgar).toarray()# 2 ye ayırna işlemi
#print(ruzgar)

#encoder: Kategorik -> Numeric
oyun= veriler.iloc[:,4:5].values#oynanmayı ayirma
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
oyun[:,-1] = le.fit_transform(veriler.iloc[:,-1])#encoding işlemi
ohe = preprocessing.OneHotEncoder()
oyun= ohe.fit_transform(oyun).toarray()# 2 ye ayırna işlemi
#print(oyun)

#dataframe oluşturma
sonuc = pd.DataFrame(data=hava, index = range(14), columns = ['sunny','overcast','rainy'])
#print(sonuc)

sonuc2 = pd.DataFrame(data = sicaklik[:,:1], index = range(14), columns = ['temperature'])
#print(sonuc2)

sonuc3=pd.DataFrame(data=nem[:,:1],index=range(14),columns=['humidity'])
#print(sonuc3)

sonuc4=pd.DataFrame(data=ruzgar[:,-1],index=range(14),columns=['windy'])
#print(sonuc4)

sonuc5=pd.DataFrame(data=oyun[:,-1],index=range(14),columns=['play'])
#print(sonuc5)



#dataframe birlestirme islemi

s=pd.concat([sonuc,sonuc2], axis=1)# sonuc ile sonuc2 birlestirme
#print(s)

s2=pd.concat([s,sonuc3], axis=1)#sonuc2 ile sonuc3 birlestirme
#print(s2)

s3=pd.concat([s2,sonuc4],axis=1)
#print(s3)

s4=pd.concat([s3,sonuc5],axis=1)
print(s4)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
# %33 test gerisi train olmak üzere ve y_tes olarak play olarak seçtik.
x_train, x_test,y_train,y_test = train_test_split(s3,sonuc5,test_size=0.33, random_state=0)


# burda bu sefer nem değişkenini y değişkeni haline getirdik
# boyun sağındaki ve solundaki verileri birleştirerek x değişkeni haline getirdik
# x_test deki verilere dayanarak y_testi tahmin etmeye çalıştık.
#tahmin ettiğimiz değerler y_predict içinde saklanıyor.
#makinemizi ise 66 test datası 33 ü train datası olarak ayırdık. Makinemiz %33 train datadan y_test datasını tahmin etmeye çalıştı

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

nem = s4.iloc[:,4:5].values #nemi ayırıyoruz

sol = s4.iloc[:,:4]#nemin solundakileri ayırıyoruz.
sag = s4.iloc[:,5:]#nemin sağındakilerini ayırıyoruz.


veri = pd.concat([sol,sag],axis=1) #sol ve sağı birleştiriyoruz.
# amaç veri kümesinin içinden nemi çıkarmak.
print(veri)

x_train, x_test,y_train,y_test = train_test_split(veri,nem,test_size=0.33, random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)
print(y_pred)

# amacımız şablon işa etmek
# Geriye eleme yöntemini kullanmak
import statsmodels.api  as sm

#1 lerden oluşan bir matris eklemek içinn:
X=np.append(arr=np.ones((14,1)).astype(int), values=veri, axis=1)
X_liste=veri.iloc[:,[0,1,2,3,4,5]].values
X_liste=np.array(X_liste,dtype=float)
model=sm.OLS(nem,X_liste).fit()
print(model.summary())
#kodu buraya kadar çalıştırdığımızda P değeri en yüksek çıkan x5(0.717) değerini Geriye eleme yöntemi ile eliyecez.

#P değeri en yüksek olan değer windy olduğu için onu çıkaracaz.




sol = veri.iloc[:,:4]#windy solundakileri ayırıyoruz.
sag = veri.iloc[:,5:]#windy sağındakilerini ayırıyoruz.
veri = pd.concat([sol,sag],axis=1) #sol ve sağı birleştiriyoruz.
# amaç veri kümesinin içinden windy çıkarmak. (P dereği yüksek olduğu için)


#windy i eliyoruz. 
X_liste=veri.iloc[:,[0,1,2,3,4]].values
X_liste=np.array(X_liste,dtype=float)
model=sm.OLS(nem,X_liste).fit()
print(model.summary())

#şimdi ise daha iyi tahminler yapabilmesi için t_train ve x_testden windy i çıkarıcaz.

#x_testden çıkarma
sol = x_test.iloc[:,:4]#windy solundakileri ayırıyoruz.
sag = x_test.iloc[:,5:]#windy sağındakilerini ayırıyoruz.
x_test = pd.concat([sol,sag],axis=1) #sol ve sağı birleştiriyoruz.
# amaç veri kümesinin içinden windy çıkarmak. (P dereği yüksek olduğu için)

#x_trainden çıkarma
sol = x_train.iloc[:,:4]#windy solundakileri ayırıyoruz.
sag = x_train.iloc[:,5:]#windy sağındakilerini ayırıyoruz.
x_train = pd.concat([sol,sag],axis=1) #sol ve sağı birleştiriyoruz.
# amaç veri kümesinin içinden windy çıkarmak. (P dereği yüksek olduğu için)

#tekrar sistemi x_train ve x_test ile eğitme
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#çıkardıktan sonra verilerin daha düzgün tahmin edildiği görülüyor.
#daha da doğru tahminler yapmak istesek P değeri en yüksek olan x3 ü çıkarmamız gerekir.




