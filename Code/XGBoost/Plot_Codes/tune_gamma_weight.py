import matplotlib.pyplot as plt
from matplotlib import pyplot

m = [0.0,0.1,0.2,0.3,0.4]

colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.792874 ,0.792054,0.792203,0.792507,0.792900]
y1 = [ 0.885283,0.884618,0.884551,0.885260,0.884928]

plt.plot(m,y0,'ro')
plt.plot(m,y1,'bs')

plt.xlabel('gamma')
plt.ylabel('Accuracy')

legend1 = plt.legend(['test_score', 'train_score'], loc=0)
#plt.show()
plt.savefig("tune_gamma.png")
