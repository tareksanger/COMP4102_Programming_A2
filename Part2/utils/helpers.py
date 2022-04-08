import glob
import unicodedata
import torch
import codecs
from .constants import ALL_LETTERS, N_CATEGORIES, N_LETTERS, CATEGORIES
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


'''
    Finds files of specified type
'''
def findFiles(path): return glob.glob(path)

'''
 Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
'''
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

'''
 Read a file and split into lines
 def readLines(filename):
     lines = open(filename, encoding='utf-8').read().strip().split('\n')
     return [unicodeToAscii(line) for line in lines]
'''
def readLines(filename):
    lines = codecs.open(filename, "r",encoding='utf-8', errors='ignore').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# -------- Turning Strings into Tensors --------------

'''
 Find letter index from ALL_LETTERS, e.g. "a" = 0
'''
def letterToIndex(letter): return ALL_LETTERS.find(letter)

'''
    Just for demonstration, turn a letter into a <1 x N_LETTERS> Tensor
'''
def letterToTensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

'''
 Turn a line into a <line_length x 1 x N_LETTERS>,
 or an array of one-hot letter vectors
'''
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

import random

def categoryFromOutput(output):
    _, top_i = output.topk(1)
    category_i = top_i[0].item()
    return CATEGORIES[category_i], category_i


def randomTrainingPair(x, y):
    def randomChoice(l): return l[random.randint(0, len(l) - 1)]

    category = randomChoice(y)
    line = randomChoice(x[category])
    category_tensor = torch.tensor([y.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def trainOneEpoch(model, x, y, lr, criterion):
    hidden = model.initHidden()

    model.zero_grad()

    for i in range(x.size()[0]):
        output, hidden = model(x[i], hidden)

    loss = criterion(output, y)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-lr)

    return output, loss.item()


def lossEvaluate(model, category_tensor, line_tensor, criterion):
    hidden = model.initHidden()

    for i in range(line_tensor.size()[0]):
          output, hidden = model(line_tensor[i], hidden)
            
    loss = criterion(output, category_tensor)
    loss.backward()

    return loss.item()

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)



# Just return an output given a line
def evaluate(model, input):
    hidden = model.initHidden()

    for i in range(input.size()[0]):
        output, hidden = model(input[i], hidden)

    return output


def predict(model, input, n_predictions=3):
    print('\n> %s' % input)
    with torch.no_grad():
        output = evaluate(model, lineToTensor(input))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, CATEGORIES[category_index]))
            predictions.append([value, CATEGORIES[category_index]])



'''
Input: trained model, a list of words, a list of class labels as integers
Output: The accuracy of the given model on the given input X and target y
'''
def calculateAccuracy(model, x, y):
    y_pred = []
    for input in x: 
        output = evaluate(model, input)
        guess, _ = categoryFromOutput(output)
        y_pred.append(guess)
    
    return accuracy_score(y, y_pred)


''' 
    Get the confusion Matrix for a given Model
'''
def confusionMatrix(model, x, y):
    
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(N_CATEGORIES, N_CATEGORIES)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    numCorrect = 0
    numTotal = 0

    for i in range(n_confusion):
        category, _, _, line_tensor = randomTrainingPair(x=x, y=y)
        output = evaluate(model, input=line_tensor)
        guess, guess_index = categoryFromOutput(output)
        
        if guess == category:
            numCorrect = numCorrect+1

        category_i = y.index(category)
        confusion[category_i][guess_index] += 1
        numTotal = numTotal + 1

    # Normalize by dividing every row by its sum
    for i in range(N_CATEGORIES):
        confusion[i] = confusion[i] / confusion[i].sum()


    accuracy = numCorrect/numTotal
    print(accuracy)

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + y, rotation=90)
    ax.set_yticklabels([''] + y)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()