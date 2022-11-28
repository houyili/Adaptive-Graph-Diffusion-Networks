import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-paper')

plt.figure(1, facecolor="white")
plt.cla()


x = np.arange(0, 6)


y2 = [0.8360, 0.8748, 0.8820, 0.8870, 0.8895, 0.8901]
y3 = [0.9134, 0.9382, 0.9422, 0.9452, 0.9485, 0.9505]
# y3 = [0.281183973
# ,0.281797251
# ,0.282384611
# ,0.283417338
# ,0.281402368]
# y4 = [0.3277218
# ,0.301661931
# ,0.292303897
# ,0.254873748
# ,0.225448094]
# y5 = [0.30995244
# ,0.30597273142
# ,0.297417406
# ,0.29853654
# ,0.297707296]
# y6 = [0.330987631
# ,0.311899121
# ,0.294865635
# ,0.291511633
# ,0.280604276]

# ax3 = ax2.twinx()
# ax1.bar(x, y2, fc='cornflowerblue')
# plt.plot(x, y2)
plt.plot(x, y2, label="Test set", marker='o',)
plt.plot(x, y3, label="Validation set",marker='o')
# plt.plot(x, y4, label="PDA",marker='o')
# plt.plot(x, y5, label="CD$^2$AN$_{unbias}$",marker='o',linewidth=2.5)
# plt.plot(x, y6, label="CD$^2$AN$_{bias}$",marker='o',linewidth=2.5)
# plt.plot(x, y1, 'seagreen', label = 'BaseModel Hitrate', marker='o')
# ax2.plot(x, y2, 'b-')
# ax3.plot(x, y1, 'seagreen', label = 'ww', marker='o')
plt.xlabel("Number of layers")
# plt.ylabel("Positive Score", color='cornflowerblue')

plt.ylabel("ROC-AUC")
plt.xticks(x,['1','2','3','4','5','6'])

plt.legend()

plt.grid(True)
# plt.show()
plt.savefig('./gipa_layer.png', dpi=500, bbox_inches='tight')
# plt.savefig('longtail_exp.eps', format="eps",dpi=500, bbox_inches='tight')