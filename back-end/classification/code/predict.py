from model import *
from data import *
import sys

rnn = torch.load('char-rnn-classification.pt')

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(input_line, n_predictions=3):
    input_line = input_line.lower()
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

def guessOnce(name):
    name = name.lower()
    with torch.no_grad():
        output = evaluate(lineToTensor(name))
        topv, topi = output.topk(1, 1, True)
        value = topv[0][0].item()
        category_index = topi[0][0].item()
        return all_categories[category_index], category_index

if __name__ == '__main__':
    name = ""
    for i in range(1, len(sys.argv)):
        name += sys.argv[i]
        name += ' '

    name = name[:-1].lower()
    predict(name)