import matplotlib.pyplot as plt
from matplotlib import pyplot

m = [1,2,3,4,5,6,7,8]
my_xticks = ['softmax','softplus','softsign','relu','tanh','sigmoid','hard_sigmoid','linear']
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.638786,0.635405,0.639128,0.635275,0.635295,0.637121,0.638819,0.600097]
y1 = [0.644077,0.639613,0.643507,0.641291,0.640662,0.643876,0.644681,0.603843]
plt.xticks(m, my_xticks)
plt.plot(m,y0,'ro')
plt.plot(m,y1,'bs')

plt.xlabel('activation')
plt.ylabel('Accuracy')

legend1 = plt.legend(['test_score', 'train_score'], loc='lower left')
plt.show()
