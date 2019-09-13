# coding=utf-8
import pickle as pk
import matplotlib.pyplot as plt

fr1 = open('/home/wl/Desktop/EXTD/result/uav_pr.pkl','rb')
'''
fr2 = open('/home/mint/caffe-ssd/01pr/SSD50088.93/uav_pr.pkl','rb')
fr3 = open('/home/mint/caffe-ssd/01pr/deeper-SSD89.56/uav_pr.pkl','rb')
fr4 = open('/home/mint/caffe-ssd/01pr/downsampling/uav_pr.pkl','rb')
fr5 = open('/home/mint/caffe-ssd/01pr/upsampling89.97/uav_pr.pkl','rb')
fr6 = open('/home/mint/caffe-ssd/01pr/upsampling-and-downsampling90.15/uav_pr.pkl','rb')
'''
inf1 = pk.load(fr1)
'''
inf2 = pk.load(fr2)
inf3 = pk.load(fr3)
inf4 = pk.load(fr4)
inf5 = pk.load(fr5)
inf6 = pk.load(fr6)
'''

fr1.close()
'''
fr2.close()
fr3.close()
fr4.close()
fr5.close()
fr6.close()
'''
x1=inf1['rec']
y1=inf1['prec']
'''
x2=inf2['rec']
y2=inf2['prec']
x3=inf3['rec']
y3=inf3['prec']
x4=inf4['rec']
y4=inf4['prec']
x5=inf5['rec']
y5=inf5['prec']
x6=inf6['rec']
y6=inf6['prec']
'''
plt.figure()
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR cruve')
x11=[]
y11=[]
c1=0
'''
x22=[]
y22=[]
c2=0
x33=[]
y33=[]
c3=0
x44=[]
y44=[]
c4=0
x55=[]
y55=[]
c5=0
x66=[]
y66=[]
c6=0
'''

for i, x in enumerate(x1):
    #if x>0.05 and y1[i] > 0.05:
    x11.append(x)
    y11.append(y1[i])

'''
for i in x1:
  if i>0.05 and y1[c1] > 0.05:
    x11.append(i)
    y11.append(y1[c1])
  c1 = c1+1
'''

'''
for i in x2:
  if i>0.05 and y2[c2] > 0.05:
    x22.append(i)
    y22.append(y2[c2])
  c2 = c2+1

for i in x3:
  if i>0.05 and y3[c3] > 0.05:
    x33.append(i)
    y33.append(y3[c3])
  c3 = c3+1

for i in x4:
  if i>0.05 and y4[c4] > 0.05:
    x44.append(i)
    y44.append(y4[c4])
  c4 = c4+1

for i in x5:
  if i>0.05 and y5[c5] > 0.05:
    x55.append(i)
    y55.append(y5[c5])
  c5 = c5+1

for i in x6:
  if i>0.05 and y6[c6] > 0.05:
    x66.append(i)
    y66.append(y6[c6])
  c6 = c6+1
'''
print(len(x11))
print(len(y11))
print(x11,y11)
#plt.ylim(0,1)
#plt.xlim(0,1)
plt.plot(x11,y11,color = 'tan')
#plt.plot(x1,y1,color = 'tan')
'''
plt.plot(x22,y22,color = 'm')
plt.plot(x33,y33,color = 'blue')
plt.plot(x44,y44,color = 'chartreuse')
plt.plot(x55,y55,color = 'gold')
plt.plot(x66,y66,color = 'red')
'''
plt.show()
print('AP：',inf1['ap'])
'''
print('AP：',inf2['ap'])
print('AP：',inf3['ap'])
print('AP：',inf4['ap'])
print('AP：',inf5['ap'])
print('AP：',inf6['ap'])
'''

