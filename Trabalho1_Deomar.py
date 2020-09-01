#!/usr/bin/env python
# coding: utf-8

#Importação das bibliotecas utilizadas
#Esse trabalho foi feito no jupyter notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.dates as mdates

###Importa os valores da série temporal###
serie = pd.read_csv('AirPassengers.csv') #Dataset dos passageiros mensais de linhas aéreas nos EUA
df = serie['#Passengers'] #Visualização do 'head' do dataset
serie['Time'] = pd.to_datetime(serie['Month']) #Transforma as datas para 'timestamp'

###Plot da série temporal###
fig = plt.figure(figsize=(8,6)) #Configura a imagem a ser plotada
ax = fig.add_subplot(111)
ax.plot(serie['Time'][:70], serie['#Passengers'][:70])
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylabel('Passageiros', size=14)
ax.set_xlabel('Data', size=14)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
fig.autofmt_xdate()
plt.savefig("Serie_passageiros.png", dpi=120) #Plota os dados de 1949 até 1955

###Aplicando o teste de Dickey Fuller###
Y = np.diff(serie['#Passengers']) #Diferença da série no tempo 't' com o tempo 't-1'
X = sm.add_constant(serie['#Passengers'][:-1]) #Transforma em matriz nx2 para a função de ajuste
model = sm.OLS(Y, X) #Cria a instância do modelo de ajuste por mínimos quadrados com os vetores X e Y
results = model.fit() #Ajusta o modelo
print("beta = {}".format(results.params[1])) #Printa os parâmetros

###Transforma a série em estacionária###
df = np.diff(np.log(serie['#Passengers']).rolling(2).mean()) #Transforma a série em estacionária
df = df[1:]
plt.plot(df) #Teste visual da estacionariedade da série

#Aplicando o teste de dickey fuller novamente
Y = np.diff(df) #Diferença da série no tempo 't' com o tempo 't-1'
X = sm.add_constant(df[:-1]) #Transforma em matriz nx2 para a função de ajuste
model = sm.OLS(Y, X) #Cria a instância do modelo de ajuste por mínimos quadrados com os vetores X e Y
results = model.fit() #Ajusta o modelo
print("beta = {}".format(results.params[1])) #Printa os parâmetros

#Plota a série estacionária
fig = plt.figure(figsize=(8,6)) #Configura a imagem a ser plotada
ax = fig.add_subplot(111)
ax.plot(serie['Time'][:70], df[:70])
import matplotlib.dates as mdates
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_xlabel('Data', size=14)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
fig.autofmt_xdate()
plt.savefig("Serie_passageiros_normalizada.png", dpi=120) #Plota os dados de 1949 até 1955


###Divindo a série em treino e teste###
train = df[:130]
test = df[130:]

#Cálculo da autocovariância
def acov(x, k): #Função que calcula a autocovariância para a matriz do Yule-Walker
    n = len(x) #Tamanho da série
    xm = x.mean() #Média da série
    autocov = 0
    for i in np.arange(n - k):
        autocov += (x[i + k] - xm)*(x[i] - xm) #Somatório para a autocovariância
    return autocov/n

###Cálculo dos coeficientes phi e Yule-Walker###
def calc_phi(serie, ordem): #Função que calcula os coeficientes coeficientes do modelo autoregressivo
    auto_cov = []
    for k in np.arange(ordem + 1): #Loop que calcula as autocovariâncias da matriz
        auto_cov.append(acov(serie,k)) #Lista com as autocovariâncias
    auto_vet = np.array(auto_cov)[1:] #Transforma em array para o cálculo
    matrix = np.zeros((ordem, ordem)) #Cria matriz zerada para o Yule-Walker
    for i in np.arange(ordem): #Preenche a matriz com as autocovariâncias
        for j in np.arange(ordem):
            matrix[i][j] = auto_cov[np.abs(i - j)]
    matrix_inv = np.linalg.inv(matrix) #Inverte a matriz para calcular os coeficientes
    phi = np.dot(matrix_inv, auto_vet) #Multiplica a matriz inversa vetor vetor de autocovariâncias para encontrar os coeficientes
    return phi

###Cálculo da PACF###
pacf = [] #Lista da PACF zerada
ord = 30 #Ordem para calcular a PACF
for ordem in np.arange(1,ord): #Calculo dos coeficientes de cada ordem
    phi = calc_phi(train,ordem)
    pacf.append(phi[ordem-1]) #Agrega os coeficientes do PACF

#Plot da PACF
#plt.plot(np.arange(len(pacf)), pacf, ls='--') #Plot do PACF e dos limites
#plt.plot(np.arange(ord), 1.96/np.sqrt(len(train))*np.ones(ord), color='red', ls=':')
#plt.plot(np.arange(ord), -1.96/np.sqrt(len(train))*np.ones(ord), color='red', ls=':')
#plt.plot(np.arange(ord), np.zeros(ord), color='black')
#plt.xlim(-1,30)
#plt.ylabel('PACF')
#plt.xlabel('k')
#plt.savefig('PACF.png', dpi=120)

###Cálculo da constante C###
p = 13 #Ordem escolhida na PACF para o modelo autoregressivo
xm = train.mean() #Média da série
c = (1-np.sum(phi)) * xm  #Cálculo da constante
c = float(c)
#print("c = {}".format(c)) #Printa a constante

###Predição do modelo autoregressivo###
pred = list(train) #Lista vazia para as previsões
for i in range(p): #Calculo da predição
    aux = 0 #Variável auxiliar para guardar os pontos da soma
    for j in range(p):
        aux += phi[j]*pred[len(train)-1+i-j] #Cálculo dos pontos anteriores com os coeficientes
        aux = float(aux)
    pred.append(c+aux) #Agrega os valores preditos
predicao = pred[len(pred)-p:] #Filtra os valores da predição
fig = plt.figure(figsize=(8,6)) #Configura a imagem a ser plotada
ax = fig.add_subplot(111)
ax.plot(serie['Time'][132:],predicao[:-1], label='Valores preditos')
ax.plot(serie['Time'][132:],df[130:], label='Valores reais')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_xlabel('Data', size=14)
plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12) 
plt.legend()
fig.autofmt_xdate()
plt.savefig('Predicao.png', dpi=120)