import matplotlib.pyplot as plt
from matplotlib import pyplot

m = [1,2,3,4,5,6,7]
my_xticks = ['SGD','RMSprop','Adagrad','Adadelta','Adam','Adamax','Nadam']
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.674419,0.635827,0.665113,0.672809,0.674479,0.675660,0.671398]
y1 = [0.678930,0.637702,0.668466,0.677347,0.679945,0.680916,0.675688]
plt.xticks(m, my_xticks)
plt.plot(m,y0,'ro')
plt.plot(m,y1,'bs')

plt.xlabel('optimizer')
plt.ylabel('Accuracy')

legend1 = plt.legend(['test_score', 'train_score'], loc='lower left')
plt.show()