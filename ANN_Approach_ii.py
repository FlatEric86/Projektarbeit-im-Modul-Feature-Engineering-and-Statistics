
import numpy as np
import pandas as pd
import torch, os
import torch.nn as nn
import random as rand
import matplotlib.pyplot as plt
import time as time


def fill_with_0(val_list):
    # if 5 - len(val_list) <= 0:
        # print(80*'')
        # print(5 - len(val_list))
    return val_list + [0 for i in range(5 - len(val_list))]



X = []
Y = []


#### read data and put them into a list for converting after to torch tensor object
for sub_dir in os.listdir('./__DATA__/data'):
    
    item_id = int(sub_dir)
    # print(80*'~')
    # print(item_id)
    
    for fname in os.listdir('./__DATA__/data/' + sub_dir):
        for i in range(1, len(os.listdir('./__DATA__/data/' + sub_dir))//2 + 1):
            if str(i) in fname:
                
                if 'meta_data' in fname:
                    lam = float(pd.read_csv('./__DATA__/data/' + sub_dir + '/' + fname)['lambda']) 
                    if np.isnan(lam):
                        continue
                    Y.append(float(0.1*lam))
                    # print(lam)
                if 't_data' in fname:
                    vals = list(pd.read_csv('./__DATA__/data/' + sub_dir + '/' + fname)['N items at day'])
                    # convert each value to a explicite float object to avoid pytorch problems
                    vals = [float(val) for val in vals]
                    
                    
                    # fill list with zeros to get a standard length of 5
                    X.append(fill_with_0(vals))
                    #X.append(vals[:4])
                    # print(vals)
                    
                    

X = torch.tensor(X)
Y = torch.tensor(Y)


###############################################################################

t_0 = time.time()

# check if CUDA is possible and store value
use_cuda = torch.cuda.is_available()

# define device in dependency if CUDA is available or not. 
device   = torch.device('cuda:0' if use_cuda else 'cpu')

# define number of CPU cores if device gots defined as CPU
if use_cuda == False:
    torch.set_num_threads(14)   # let one core there for os



# splitte die Trainingsdaten zu je (ca.) 50 % in tatsächlichen Trainingsdaten und Validierungsdaten
X_training = X[:len(X)//2]
X_training = X_training.to(device) 

X_valid    = X[len(X)//2:]
X_valid    = X_valid.to(device) 

Y_training = Y[:len(Y)//2]
Y_training = Y_training.to(device) 

Y_valid    = Y[len(Y)//2:]
Y_valid    =  Y_valid.to(device) 








Number_Input_Neurons  = len(X[0])                # number of input neurons
Number_Output_Neurons = 1                        # number of outpu neurons
Learning_Rate         = 0.000005                 # laerning rate
Number_Epochs         = 5000                   # number of epoches 
Model_Parameter_File  = "model_status_cuda.pt"   # file name of weight matrix


# definiere  NN-Topologie  !!! Wichtig ist, dass das Objekt als CUDA-Tensorobjekt deklariert werden muss, wenn über die GPU gerechnet werden soll
model = nn.Sequential( 
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.ReLU(),  
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.ReLU(),
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.ReLU(),  
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.ReLU(),  
            nn.Linear(Number_Input_Neurons, Number_Input_Neurons),
            nn.ReLU(),              
            nn.Linear(Number_Input_Neurons, 1)
).to(device)
 

# definiere Fehlerfunktion für Evaluierungszwecke
def err(true_Y, pred_Y, mode=None):

    err_sum = 0
    for true_val, model_val in zip(true_Y, pred_Y):
        if mode == 'abs':
            err_sum += abs(true_val - model_val)

        if mode == 'qrd':
            err_sum += (true_val - model_val)**2


    return err_sum
    
    
    
# Lädt die gespeicherten Weights, wenn Datei vorhanden
if os.path.isfile(Model_Parameter_File):
    model.load_state_dict(torch.load(Model_Parameter_File))

# Loss Function vorgeben (Mean Squared Error Loss)
criterion = torch.nn.MSELoss(reduction='sum')

# Optimizer vorgeben (SGD Stochastic Gradient Descent)  # hat sich für dieses Model als signifikant 
# schlecht gegenüber dem ADAM-Optimizer herausgestellt
#optimizer = torch.optim.SGD(model.parameters(), lr = Learning_Rate, momentum=0.5)

# ADAM Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=Learning_Rate)

# Das ist jetzt die eigentliche Ausführung des Prozesses

# Liste zum sammeln aller Modellfehler hinsichtlich sowohl der Prädiktionsfehler als auch der Fehler hinsichtlich
# der Validierungsadten, für anschließende Evaluierungszwecke
ERRs     = []
ERR_rand = []



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Optimierungsprozess des NN

# Iteriere über die vorgegebene Anzahl an Epochen
for epoch in range(Number_Epochs):


    # Vorwärts Propagierung
    Y_predicted = model(X_training).to(device)



    # Berechne Fehler und gebe ihn aus
    loss = criterion(Y_predicted, Y_training)
    print('epoch: ', epoch, ' loss: ', loss.item())

    # wende Gradientenabstiegsmethode auf den Optimizer an
    optimizer.zero_grad()

    # Führe Modellfehler rückwertig auf das Modell zurück
    loss.backward()
    model.float()

    # Update the parameters
    optimizer.step()


    ###### Evolution

    Y_valid_model = model(X_valid)

    err_pred_vs_training = float(err(Y_training, Y_predicted, mode='abs')) 
    err_pred_vs_valid    = float(err(Y_valid, Y_valid_model, mode='abs'))

    MAE_training         = err_pred_vs_training/len(Y_training)
    MAE_valid            = err_pred_vs_valid/len(Y_valid)

    ERRs.append([MAE_training, MAE_valid])

    rand_idx = rand.randint(0,len(Y_training)-1)

    rand_true_y = float(Y_valid[rand_idx])
    rand_pred_y = float(Y_valid_model[rand_idx])

    ERR_rand.append(abs(rand_true_y-rand_pred_y))

    print(80*'~')
    # print(epoch)
    # print('Training_err_abs  : ', err_pred_vs_training)
    # print('Validation_err_abs: ', err_pred_vs_valid)
    # #print('-------------------')
    # #print('Training_err_qrd  : ', err_pred_vs_training_qrd)
    # #print('Validation_err_qrd: ', err_pred_vs_valid_qrd)
    # print('-------------------')
    # print('random_valid:')
    print('True y            :', rand_true_y)
    print('pred y            :', rand_pred_y) 
    # print('abs_err           :', abs(rand_true_y-rand_pred_y))



# # Plotte Ergebnisse zur Validierung des Modells

epoch_n = 1
flag    = 0
epochs = []


for sublist in ERRs:

    if (epoch_n-1) % 100 == 0:

        if flag == 0:

            plt.scatter(epoch_n, sublist[0], color='green', label='Training')
            plt.scatter(epoch_n, sublist[1], color='blue', label='Validation')
            #plt.scatter(epoch_n, ERR_rand[epoch_n-1], color='red', label='rand_validation')
            epochs.append(epoch_n)
            flag = 1
            continue

        else:
            plt.scatter(epoch_n, sublist[0], color='green')
            plt.scatter(epoch_n, sublist[1], color='blue')
            epochs.append(epoch_n)
    


    epoch_n += 1

#plt.plot(epochs, ERR_rand, color='red', alpha=0.5)
print(80*'=' + '\n')
print(str(time.time()-t_0))


plt.legend()
plt.show()

torch.save(model.state_dict(), Model_Parameter_File)

