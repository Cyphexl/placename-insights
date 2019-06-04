import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data import *
from predict import *

confusion = torch.zeros(n_categories, n_categories)
n_confusion = 4000

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor

for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    guess, guess_i = guessOnce(line)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1
    
print("Confusion Matrix:")
for i in range(n_categories):
    print(confusion[i])
    confusion[i] = confusion[i] / confusion[i].sum()

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()