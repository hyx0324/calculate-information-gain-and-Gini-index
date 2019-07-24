import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io


data_str = output = io.StringIO('''编号,色泽,根蒂,敲声,纹理,脐部,触感,密度,含糖率,好瓜
1,青绿,蜷缩,浊响,清晰,凹陷,硬滑,0.697,0.46,是  
2,乌黑,蜷缩,沉闷,清晰,凹陷,硬滑,0.774,0.376,是  
3,乌黑,蜷缩,浊响,清晰,凹陷,硬滑,0.634,0.264,是  
4,青绿,蜷缩,沉闷,清晰,凹陷,硬滑,0.608,0.318,是  
5,浅白,蜷缩,浊响,清晰,凹陷,硬滑,0.556,0.215,是  
6,青绿,稍蜷,浊响,清晰,稍凹,软粘,0.403,0.237,是  
7,乌黑,稍蜷,浊响,稍糊,稍凹,软粘,0.481,0.149,是  
8,乌黑,稍蜷,浊响,清晰,稍凹,硬滑,0.437,0.211,是  
9,乌黑,稍蜷,沉闷,稍糊,稍凹,硬滑,0.666,0.091,否  
10,青绿,硬挺,清脆,清晰,平坦,软粘,0.243,0.267,否  
11,浅白,硬挺,清脆,模糊,平坦,硬滑,0.245,0.057,否  
12,浅白,蜷缩,浊响,模糊,平坦,软粘,0.343,0.099,否  
13,青绿,稍蜷,浊响,稍糊,凹陷,硬滑,0.639,0.161,否  
14,浅白,稍蜷,沉闷,稍糊,凹陷,硬滑,0.657,0.198,否  
15,乌黑,稍蜷,浊响,清晰,稍凹,软粘,0.36,0.37,否  
16,浅白,蜷缩,浊响,模糊,平坦,硬滑,0.593,0.042,否  
17,青绿,蜷缩,沉闷,稍糊,稍凹,硬滑,0.719,0.103,否  ''')

data = pd.read_csv(data_str)
data.set_index('编号', inplace=True)
print(data)


def entropy(data, att_name):
    '''
    :param data: 数据集
    :param att_name: 属性名
    :return: entropy
    '''

    levels = data[att_name].unique()

    ent = 0
    for lv in levels:
        pi = sum(data[att_name]==lv) / data.shape[0]
        ent += pi * np.log2(pi)

    return -ent


print('\n好瓜的信息熵：', entropy(data, '好瓜'))


def conditional_entropy(data, xname, yname):
    xs = data[xname].unique()
    ys = data[yname].unique()
    p_x = data[xname].value_counts() / data.shape[0]
    ce = 0

    for x in xs:
        ce += p_x[x] * entropy(data[data[xname]==x], yname)

    return ce


print('\n色泽条件下，好瓜的信息熵：', conditional_entropy(data, '色泽', '好瓜'))


def gain(data, xname, yname):
    ent = entropy(data, yname)
    ce = conditional_entropy(data, xname, yname)
    return ent - ce


print('\n色泽的引入导致的信息增益：', gain(data, '色泽', '好瓜'))

gain_list = []
print('\n全部特征的信息增益：')
for name in data.columns[:-3]:
    name_gain = gain(data, name, '好瓜')
    gain_list.append(name_gain)
    print(name, name_gain)


def gini_index(data, xname, yname):
    xs = data[xname].unique()
    p_x = data[xname].value_counts() / data.shape[0]

    gi_index = 0

    for x in xs:
        data_x = data[data[xname]==x]
        ys = data[yname].unique()
        gi = 1

        for y in ys:
            if y not in data_x[yname].value_counts():
                continue

            gi -= np.square(data_x[yname].value_counts()[y] / data_x.shape[0])

        gi_index += p_x[x] * gi

    return gi_index


#print(gini_index(data, '根蒂', '好瓜'))

gini_list = []
print('\n全部的基尼指数：')
for name in data.columns[:-3]:
    name_gi = gini_index(data, name, '好瓜')
    gini_list.append(name_gi)
    print(name, name_gi)

x = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
plt.plot(x, gain_list)
plt.plot(x, gini_list)
plt.show()