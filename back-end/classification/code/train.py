import torch
from data import *
from model import *
import random
import time
import math

n_hidden = 128
n_epochs = 80000
print_every = 500
plot_every = 100
learning_rate = 0.003

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingPair():
    # category = randomChoice(all_categories)
    # line = randomChoice(category_lines[category])
    tuple = randomChoice(all_lines)
    line = tuple[0]
    category = tuple[1]
    category_tensor = Variable(torch.LongTensor([all_categories.index(category)]))
    line_tensor = Variable(lineToTensor(line))
    return category, line, category_tensor, line_tensor

rnn = RNN(n_letters, n_hidden, n_categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    outputt = 0
    loss = 0
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        outputt, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(outputt, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return outputt, loss.item()



current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

for epoch in range(1, n_epochs + 1):
    category, line, category_tensor, line_tensor = randomTrainingPair()
    try:
        output, loss = train(category_tensor, line_tensor)
    except:
        continue
    current_loss += loss

    if epoch % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, timeSince(start), loss, line, guess, correct))

    if epoch % plot_every == 0:
        averageLoss = current_loss / plot_every
        print("Average loss: %f" % averageLoss)
        all_losses.append(averageLoss)
        current_loss = 0

torch.save(rnn, 'char-rnn-classification.pt')

