import matplotlib.pyplot as plt
from matplotlib import pyplot

x = [0.3,0.5,0.8,1]
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y = [0.673340,0.679201,0.682346,0.683263]

z = [0.678779,0.686036,0.689963,0.691155]

plt.plot(x,y,'ro')
plt.plot(x,z,'bs')

plt.xlabel('feature_fraction')
plt.ylabel('Accuracy')

legend1 = plt.legend(['test_score', 'train_score'], loc='lower right')
plt.show()