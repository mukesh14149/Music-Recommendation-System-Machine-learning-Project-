import matplotlib.pyplot as plt
from matplotlib import pyplot

x1 = [0.01,0.1,1.0,10.0]
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y = [0.646478,0.683076,0.696336,0.417627]
z = [0.647215,0.690700,0.748241,0.418213]

plt.plot(x1,y,'ro')
plt.plot(x1,z,'bs')

plt.xlabel('learning_rate')
plt.ylabel('Accuracy')

legend1 = plt.legend(['test_score', 'train_score'], loc='lower left')
plt.show()
