'''
1.confusion matrix

混淆矩阵
          1     0
     1   TP     FN

     0   FP     TN

1.准确率 ：   TP+TN/(TP+FN+FP+TN)
2.精准率 ：   TP/(TP+FP)
3.RECALL:    TP/(TP+FN)
4.F1 SCORE : 调和平均数    F1 = 2* (PRE*ACC/(PRE+ACC))
5.F beta Score :      F beta =(1+beta^2 )*(PRE*ACC/(beta^2*PRE+ACC)) beta [ 0 +∞）
1 为F1 SCORE    >1  趋向recall   1< 趋向 精准率
6.ROC曲线


FPR =  FP/N

TPR  = TP/P
通过阈值的更改 ，分别计算出 FPR TPR 画出ROC曲线


测试集中的正负样本的分布变换的时候，ROC曲线能够保持不变
7.AUC  AUC(Area under Curve)：Roc曲线下的面积，介于0.1和1之间。Auc作为数值可以直观的评价分类器的好坏，值越大越好。

8.PR曲线
x:recall
y:precision
'''

from sklearn.metrics import accuracy_score

from sklearn.metrics import  precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import  fbeta_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve

'''
https://blog.csdn.net/u014264373/article/details/80487766


y 就是标准值，scores 是每个预测值对应的阳性概率，比如0.1就是指第一个数预测为阳性的概率为0.1，很显然，y 和 socres应该有相同多的元素，都等于样本数。pos_label=2 是指在y中标签为2的是标准阳性标签，其余值是阴性。
所以在标准值y中，阳性有2个，后两个；阴性有2个，前两个。

接下来选取一个阈值计算TPR/FPR,阈值的选取规则是在scores值中从大到小的以此选取，于是第一个选取的阈值是0.8

scores中大于阈值的就是预测为阳性，小于的预测为阴性。所以预测的值设为y_=(0,0,0,1),0代表预测为阴性，1代表预测为阳性。可以看出，真阴性都被预测为阴性，真阳性有一个预测为假阴性了。

FPR = FP / (FP+TN) = 0 / 0 + 2 = 0

TPR = TP/ (TP + FN) = 1 / 1 + 1 = 0.5

thresholds = 0.8


'''

x_pred = [1,1,0,1,1,1]
x_true = [1,0,1,0,1,1]
'''
以precision为例，P表示二分类时精确率的计算结果

macro：不考虑类别数量，不适用于类别不均衡的数据集，其计算方式为： 各类别的P求和/类别数量

weighted:各类别的P × 该类别的样本数量（实际值而非预测值）/ 样本总数量

'''

print("精度 %f" %(accuracy_score(x_true,x_pred)))
print(precision_score(x_true,x_pred,average='binary'))
print(precision_score(x_true,x_pred,average='macro'))
print(precision_score(x_true,x_pred,average='weighted'))
print(recall_score(x_true,x_pred))
print(f1_score(x_true,x_pred))
print(fbeta_score(x_true,x_pred,beta=2))





