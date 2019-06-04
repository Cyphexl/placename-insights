from models.data import *

rnn = torch.load('./models/char-rnn-classification.pt')
all_categories=['EastAsia', 'S&SEAsia', 'EnUsAuNz', 'Latinos', 'Arabics', 'WEurope', 'EEurope', 'Oceania', 'SSAfrica']

def evaluate(line_tensor):
    hidden = rnn.initHidden()
    output = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(input_line, n_predictions=3):
    try:
        input_line = input_line.lower()
        with torch.no_grad():
            output = evaluate(lineToTensor(input_line))
            topv, topi = output.topk(n_predictions, 1, True)
            predictions = []

            for i in range(n_predictions):
                value = round(topv[0][i].item(),4)
                category_index = topi[0][i].item()
                predictions.append([value, all_categories[category_index]])

            return predictions
    except:
            return False

def guessOnce(name):
    name = name.lower()
    with torch.no_grad():
        output = evaluate(lineToTensor(name))
        topv, topi = output.topk(1, 1, True)
        value = topv[0][0].item()
        category_index = topi[0][0].item()
        return all_categories[category_index], category_index


predict("beijing")