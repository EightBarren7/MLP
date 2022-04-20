from tqdm import tqdm
from model import MLP
import numpy as np
from dataloader import dataloader


num_epochs = 100
batch_size = 256

train_iter, test_iter, len_train_data, len_test_data = dataloader(batch_size)

model = MLP(input_size=784, batch_size=batch_size, num_classes=10, lr=0.001, hidden_layer_sizes=(256,),
            activation='tanh')
model.parameter()

for epoch in range(num_epochs):
    acc = 0
    with tqdm(train_iter, unit='batch') as tepoch:
        for data, label in tepoch:
            tepoch.set_description(f"Epoch {epoch + 1} train")
            if data.shape[0] < batch_size:
                break
            data = data.numpy()
            label = label.numpy()
            outputs = model.forward(data)
            acc += (outputs.argmax(1) == label).sum() / len_train_data
            model.backward(np.eye(10)[label])
            tepoch.set_postfix(acc=acc)
test_acc = 0
with tqdm(test_iter, unit='batch') as tepoch:
    for data, label in tepoch:
        tepoch.set_description(f"Test")
        if data.shape[0] < batch_size:
            break
        data = data.numpy()
        label = label.numpy()
        outputs = model.forward(data)
        test_acc += (outputs.argmax(1) == label).sum() / len_test_data
        tepoch.set_postfix(acc=test_acc)
print(f"Accuracy: {test_acc}")
