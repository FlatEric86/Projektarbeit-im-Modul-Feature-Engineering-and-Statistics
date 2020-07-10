
import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import random as rand
import os, shutil
from gradient_des import fit_exp as fit_func
from gradient_des import forward_model as model
from sklearn.metrics import r2_score




plt.style.use('ggplot')

if os.path.isdir('./__DATA__/plots') == False:
    os.makedirs('./__DATA__/plots')
else:
    shutil.rmtree('./__DATA__/plots')
    os.makedirs('./__DATA__/plots')

if os.path.isdir('./__DATA__/data') == False:
    os.makedirs('./__DATA__/data')
else:
    shutil.rmtree('./__DATA__/data')
    os.makedirs('./__DATA__/data')


def find_descending_sequences(pd_series):
    '''
    Funktion zum Finden von absteigenden Sequenzen in einer Artikelspalte.
    Als absteigende Sequenz soll ein Teilvektor mit den Elementen 
    (n, n+1, n+2,...m) gelten, wenn von einem ersten Element, welches ungleich
    0 sein soll, jedes folgende Element kleiner als sein vorhergehendes Element
    ist und ungleich der 0. Zudem soll eine Sequenz mindestens aus 3 Folgeelementen 
    bestehen.
    Rückgabewert ist eine Liste mit Tuplen welche jeweils den Anfangsindex und 
    Endindex des Teilvektors aus der Pandas Series enthalten.
    '''

    INDIZES = []
    
    a = 0
    while True:
        
        b = a+1
        for i in range(a, len(pd_series)):
            if pd_series[i] == 0:
                a = i+1
                break

            if pd_series[i+1] < pd_series[i] and pd_series[i+1] != 0:
                b += 1
                
            else:
                if (b - a) >= 3: # 2. Prämisse -> Die Sequenz soll mindestens 3 Elemente enthalten
                    INDIZES.append((a, b))
                    a = b
                    break
                else:
                    a = b
                    break
         
        if a == len(pd_series)-1:
            break
            
    return INDIZES
                
                  


    
 
   
###############################################################################  
   
  

df = pd.read_csv('./merged_data.csv', delimiter=';')





# filtere nach Artikeln die mindestens 7 Verkaufs-Cluster enthalten

counter = 0




ALL_R_2 = []
LAMBDAS = []
for i in range(1, df.shape[1]):

    desc_sequ = find_descending_sequences(df.iloc[:,i])
    n_cluster = len(desc_sequ)
    
    series_ = df.iloc[:,i]
    
    
    if n_cluster >= 7:

        os.mkdir('./__DATA__/plots/' + str(df.columns[i]))
        os.mkdir('./__DATA__/data/' + str(df.columns[i]))

        counter += 1

        
        
        print(80*'~' + '\n')
        print('Index Tuples : ', desc_sequ)
        print('desc_sequ', series_)
        
        k = 1
        for cluster in desc_sequ:

            


            A_0 = series_[cluster[0]]
            
            print(series_[cluster[0]: cluster[1]])
            
            Y_training = series_[cluster[0]: cluster[1]]
            coef, ERR, N = fit_func([i for i in range(len(Y_training))], Y_training, A_0, 0.0001, 0.000001)
            

            plt.scatter([i for i in range(len(Y_training))], Y_training, color='blue', label='Resell Data')
            
            
            T_model = np.arange(0, (len(Y_training)-1 + 0.1), 0.1)




            # Bestimmtheitsmaß und Plot
            try:
                R_2 = r2_score(Y_training, [model(t, A_0, coef) for t in range(len(Y_training))])
                plt.plot(T_model, [model(t, A_0, coef) for t in T_model], color='green', label= 'Regression Model | ' + '$R^2 = $' + str(round(R_2, 6))
                + '\n$\lambda$ = ' + str(round(coef, 2))
                )
            except:
                R_2 = np.nan
                plt.plot(T_model, [model(t, A_0, coef) for t in T_model], color='green', label= 'Regression Model | ' + '$R^2 = $' + 'unbestimmt'
                + '\n$\lambda$ = ' + str(round(coef, 2))
                )
            
            ALL_R_2.append(R_2)
            LAMBDAS.append(coef)


            ### write data
            
            # trainingsdata
            trainings_data = pd.DataFrame({'date':df['date'][cluster[0]: cluster[1]], 'N items at day' : Y_training})
            trainings_data.to_csv('./__DATA__/data/' + str(df.columns[i]) + '/sequ_' + str(k) + 't_data.csv', index=False)

            # meta data
            meta_data = pd.DataFrame({'lambda':[coef], 'r_2':[R_2]})
            meta_data.to_csv('./__DATA__/data/' + str(df.columns[i]) + '/sequ_' + str(k) + 'meta_data.csv', index=False)


            plt.title('Sell-Sequence# ' + str(k) + '\nof Item ' + df.columns[i])
            plt.xlabel('Date of Sale')

            plt.xticks([i for i in range(len(df['date'][cluster[0]: cluster[1]]))], df['date'][cluster[0]: cluster[1]])

            plt.ylabel('Number of sold Items')
            plt.legend()

            plt.savefig('./__DATA__/plots/' + str(df.columns[i]) + '/' + str(k) + '.png', dpi=300 )
            plt.close()
            #plt.show()
            
            
            k += 1
        
        
    #testing    
    # if counter == 1:
        # break
     

        

### Histogram of R^2 Values
        
fig,ax = plt.subplots(1, 2)

discrt = np.arange(0.7,1,0.05)

ax[0].hist(ALL_R_2, bins = discrt, color='green', alpha=0.8)
ax[0].set_title("Histogram of $R^2$ Values\nover all Sell-Sequences", size=8)
ax[0].set_xticks(discrt)
ax[0].set_xlabel('$R^2$')
ax[0].set_ylabel('Frequency')

ax[1].hist(LAMBDAS, bins = np.arange(0,max(LAMBDAS),0.1), color='blue', alpha=0.8)
ax[1].set_title("Histogram of all $\lambda$ Values\nover all Sell-Sequences", size=8)
ax[1].set_xlabel('$\lambda$')
ax[1].set_ylabel('Frequency')
plt.savefig('./__DATA__/r_2_plot.png', dpi=300)
#plt.show()

    
        

        
        
print(len(LAMBDAS))
print(len(ALL_R_2))
        
        
      
        
        
        
        
        
        
        
######################### Debugging/Testing       
        
        
# A_0 = 25


# T          = np.arange(0, 10, 3)
# Y_training = [model(t, A_0, 0.2) + 1*rand.uniform(-1, 1) for t in T]
 
 
# coef, ERR, N = fit_func(T, Y_training, A_0, 0.0001, 0.000001)

# print('Coef   = ', coef)
# print('ERR    = ', ERR)
# print('N_iter = ', N)

 
# T_mod = np.arange(0, 10, 0.01)   
# Y_mod = [model(t, A_0, coef) for t in T_mod]
    
    
# plt.scatter(T, Y_training, color='blue')
    
# plt.plot(T_mod, Y_mod, color='green')
    
    
# plt.show()

     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
