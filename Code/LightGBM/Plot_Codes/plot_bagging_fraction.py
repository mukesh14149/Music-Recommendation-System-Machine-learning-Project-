import matplotlib.pyplot as plt
from matplotlib import pyplot

bagging_fraction = [0.3,0.5,0.8]
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y = [0.679688,0.681938,0.683330]

z = [0.688457,0.690256,0.691085]

plt.plot(bagging_fraction,y,'ro')
plt.plot(bagging_fraction,z,'bs')

plt.xlabel('bagging_fraction')
plt.ylabel('Accuracy')

legend1 = plt.legend(['test_score', 'train_score'], loc='lower right')
plt.show()
