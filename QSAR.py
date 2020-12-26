import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import from_numpy

xy = np.loadtxt('./data/qsar_aquatic_toxicity.csv', delimiter=';', dtype=np.float32)
data_len = len(xy)
test_data_len = int(data_len * 0.1)
train_data_len = data_len - test_data_len
test_data_idx = -1 * test_data_len

# Split Data into train(90%) and test(10%) data
x_train = from_numpy(xy[0:train_data_len, 0:-1])
y_train = from_numpy(xy[0:train_data_len, [-1]])
x_test = from_numpy(xy[test_data_idx:, 0:-1])
y_test = from_numpy(xy[test_data_idx:, [-1]])
print(f'X\'s shape: {x_train.shape} | Y\'s shape: {y_train.shape}')
print(f'X\'s shape: {x_test.shape} | Y\'s shape: {y_test.shape}')
torch.manual_seed(1)


# our model
model = nn.Linear(8,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 


# Training
nb_epochs = 100000
for epoch in range(nb_epochs+1):

    # Make prediction
    # model(x_train) is equal to model.forward(x_train)
    prediction = model(x_train)

    # compute cost
    cost = F.mse_loss(prediction, y_train) # <== Mean squared error

    # Gradient Descent Part
    # Initialize the gradient to zero
    optimizer.zero_grad()
    # Compute the gradient using the cost information
    cost.backward()
    # Update weights and bias
    optimizer.step()

    if epoch % 10000 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
        ))


# Testing

for i in range(len(x_test)):
    y_pred = model(x_test[i])
    y_pred_val = y_pred[0]
    y_test_val = y_test[i][0]
    percentage = abs((y_test_val - y_pred_val)/y_pred_val*100)
    
    print(f"Test{i+1} Prediction: {y_pred[0]}, Real: {y_test[i][0]} ({percentage})")






