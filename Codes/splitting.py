def split(x, split_val):
    split_index = int(len(x) * split_val)
    x_train = x[:split_index]
    x_test = x[split_index:]
    return x_train, x_test

def sort_by_date(x, y, dates):
    pairs = []
    for i in range(len(dates)):
        pairs.append((dates[i], i))
    pairs.sort()
    order = []
    for pair in pairs:
        order.append(pair[1])

    x_sorted = [x[i] for i in order]
    y_sorted = [y[i] for i in order]
    dates_sorted = [dates[i] for i in order]
    return x_sorted, y_sorted, dates_sorted

def accuracy(real, predict):
    correct = 0
    for i in range(len(real)):
        if real[i] == predict[i]:
            correct += 1
    return correct / len(real) if real else 0


def countmetrics(real, predict):
    TP = FP = FN = TN = 0
    for i in range(len(real)):
        if real[i] == 1 and predict[i] == 1:
            TP += 1
        elif real[i] == 0 and predict[i] == 0:
            TN += 1
        elif real[i] == 0 and predict[i] == 1:
            FP += 1
        else:
            FN += 1

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)
   
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    confusion = [[TN, FP], [FN, TP]] 
    return precision, recall, f1, confusion


#Tests
data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
split_val = .75
train, test = split(data, split_val)
print(train)
print(test) 

real = [0, 0, 0]
predict = [0, 0, 1]
print()
print(accuracy(real, predict))     
print(countmetrics(real, predict))   
