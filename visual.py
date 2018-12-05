import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt


path = sys.argv[0]

data = pd.read_csv(path)

data['imp_prod'] = data['importance_feature'] * data['importance_visit']

test_data = data.groupby(['visit','feature']).agg({'imp_prod':'mean'})

test_data = test_data.reset_index()

test_data = test_data[test_data['imp_prod']>0.65]

test_data['revisit'] = np.floor(test_data['visit']/5)*5

test_data = test_data.groupby(['revisit','feature']).agg({'imp_prod':'mean'})

test_data = test_data.reset_index()

temp_dic = {}
colour_list = []
for f in test_data['feature']:
    if f not in temp_dic:
        temp_dic[f]=len(temp_dic)*10
    colour_list.append(temp_dic[f])

fig, ax = plt.subplots()
ax.scatter(test_data['revisit'].values, test_data['imp_prod'].values, c=colour_list)
ax.set_xlabel('Visit(s)')
ax.set_ylabel('Importance')
ax.set_title('Feature importance')


for i, txt in enumerate(test_data['feature'].values):
    ax.annotate(txt, (test_data['revisit'].values[i], test_data['imp_prod'].values[i]))

fig.savefig('graph.png', dpi=300)