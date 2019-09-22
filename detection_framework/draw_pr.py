# coding=utf-8
import pickle as pk
import matplotlib.pyplot as plt

fr1 = open('/home/wl/Desktop/EXTD/result/uav_pr.pkl','rb')
inf1 = pk.load(fr1)
fr1.close()
x1=inf1['rec']
y1=inf1['prec']

'''
fr2 = open('/home/wl/Desktop/EXTD/result/uav_pr.pkl','rb')
inf2 = pk.load(fr2)
fr2.close()
x2=inf2['rec']
y2=inf2['prec']
'''
plt.figure()
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR cruve')

'''
for i, x in enumerate(x1):
    #if x>0.05 and y1[i] > 0.05:
    x11.append(x)
    y11.append(y1[i])
'''

#plt.ylim(0,1)
#plt.xlim(0,1)
#plt.plot(x11,y11,color = 'tan')
plt.plot(x1,y1,color = 'tan')
#plt.plot(x2,y2,color = 'blue')

plt.show()
print('AP：',inf1['ap'])

"""
import matplotlib.pyplot as plt
x = fp
y = tp
plt.figure()
plt.xlabel('tp')
plt.ylabel('fp')
plt.title('ROC cruve')
plt.plot(x,y,color = 'tan')
plt.show()
rec = tp / float(npos)
"""


