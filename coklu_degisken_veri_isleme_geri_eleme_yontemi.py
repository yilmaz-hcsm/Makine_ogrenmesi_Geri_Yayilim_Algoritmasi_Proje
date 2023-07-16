# -*- coding: utf-8 -*-


#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('veriler.csv')
print(veriler)

#ülke için encoder uygulaması
#encoder: Kategorik -> Numeric
ulke = veriler.iloc[:,0:1].values#ulkeyi ayrma
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0])#encoding işlemi
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()#3 e ayırma
print(ulke)


Yas = veriler.iloc[:,1:4].values
print(Yas)

#encoder: Kategorik -> Numeric
c= veriler.iloc[:,-1:].values#cinsiyeti ayırma
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
c[:,-1] = le.fit_transform(veriler.iloc[:,-1])#encoding işlemi
ohe = preprocessing.OneHotEncoder()
c= ohe.fit_transform(c).toarray()# 2 ye ayırna işlemi
print(c)


#numpy dizileri dataframe donusumu
sonuc = pd.DataFrame(data=ulke, index = range(22), columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range(22), columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data = c[:,:1], index = range(22), columns = ['cinsiyet'])
print(sonuc3)# sadece 1 kolonu al (kukla)

#dataframe birlestirme islemi
s=pd.concat([sonuc,sonuc2], axis=1)# sonuc ile sonuc2 birlestirme
print(s)

s2=pd.concat([s,sonuc3], axis=1)#sonuc2 ile sonuc3 birlestirme
print(s2)


#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)



# burda bu sefer boy değişkenini y değişkeni haline getirdik
# boyun sağındaki ve solundaki verileri birleştirerek x değişkeni haline getirdik
# x_test deki verilere dayanarak y_testi tahmin etmeye çalıştık.
#tahmin ettiğimiz değerler y_predict içinde saklanıyor.
#makinemizi ise 66 test datası 33 ü train datası olarak ayırdık. Makinemiz %33 train datadan y_test datasını tahmin etmeye çalıştı
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:,3:4].values
print(boy)
sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]

veri = pd.concat([sol,sag],axis=1)

x_train, x_test,y_train,y_test = train_test_split(veri,boy,test_size=0.33, random_state=0)


r2 = LinearRegression()
r2.fit(x_train,y_train)

y_pred = r2.predict(x_test)


# amacımız şablon işa etmek
# Geriye eleme yöntemini kullanmak
import statsmodels.api  as sm

#1 lerden oluşan bir matris eklemek içinn:
X=np.append(arr=np.ones((22,1)).astype(int), values=veri, axis=1)

X_liste=veri.iloc[:,[0,1,2,3,4,5]].values
X_liste=np.array(X_liste,dtype=float)
model=sm.OLS(boy,X_liste).fit()
print(model.summary())
#kodu buraya kadar çalıştırdığımızda P değeri en yüksek çıkan x5(0.717) değerini Geriye eleme yöntemi ile eliyecez.

#x5 i eliyoruz. 
X_liste=veri.iloc[:,[0,1,2,3,5]].values
X_liste=np.array(X_liste,dtype=float)
model=sm.OLS(boy,X_liste).fit()
print(model.summary())

# kodu buraya kadar çalıştırdığımızda P değeri en yüksek x5(0.031) çıkıyor.
# bu hata kabul değerinin (0.05) aldında olduğu için yine eleme işlemi yapmıyoruz.






