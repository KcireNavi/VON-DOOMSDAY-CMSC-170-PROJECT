def split(x, split):
    split_index = int(len(x) * split)
    x_train = x[split_index:] #The one used for training whihc is the first 80% oldest 
    x_data = x[:split_index] #the one used for testing which is the newest being the last 2-%
    return x_data, x_train

def accuracy(real, guessed):
    correct = 0
    for i in range(len(real)):
        if real[i] == guessed[i]:
            correct += 1
    return correct / len(real) if real else 0 


#Tests
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
split_val = .75
train, test = split(data, split_val)
print(train)
print(test)