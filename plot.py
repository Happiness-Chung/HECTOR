import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

folder_path = 'C:/Users/user/Desktop/code/HECTOR/History/2024-02-13_8H/'

f1 = open(folder_path + '/training_performance.txt', 'r')
f2 = open(folder_path + '/validation_performance.txt', 'r')

accs1 = f1.readlines()
accs2 = f2.readlines()

acc1 = []
acc2 = []


# Loss
x = list(range(200))
for i in range(len(accs1)):
    acc1.append(float(accs1[i].strip().split(',')[3]))
    acc2.append(float(accs2[i].strip().split(',')[3]))

sns.set_style('darkgrid')
sns.lineplot(acc1, label='Training', color='blue', alpha=0.3, linestyle=':')
sns.lineplot(acc2, label='Validation', color='red', alpha=0.3)
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.savefig(folder_path + '/Recall.png')