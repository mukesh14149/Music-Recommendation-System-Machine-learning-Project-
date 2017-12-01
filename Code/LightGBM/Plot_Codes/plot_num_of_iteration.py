import matplotlib.pyplot as plt
from matplotlib import pyplot

m = [200,300,500,700,900,1100]
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.682636,0.690743,0.701197,0.707506,0.711894,0.715404]
y1 = [0.690447,0.702490,0.720281,0.734249,0.745627,0.755743]
plt.plot(m,y0,'ro')
plt.plot(m,y1,'bs')

plt.xlabel('num_of_iteration')
plt.ylabel('Accuracy')

legend1 = plt.legend(['test_score', 'train_score'], loc='lower left')
plt.show()