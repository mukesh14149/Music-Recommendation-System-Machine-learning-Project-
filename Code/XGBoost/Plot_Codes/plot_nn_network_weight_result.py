import matplotlib.pyplot as plt
from matplotlib import pyplot

m = [1,2,3,4,5,6,7,8]
my_xticks = ['uniform','lecun_uniform','normal','zero','glorot_normal','glorot_uniform','he_normal','he_uniform']
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.676085,0.675320,0.676188,0.500000,0.674477,0.675872,0.673856,0.675496]
y1 = [0.681393,0.680083,0.681271,0.500000,0.679152,0.680424,0.679544,0.681081]
plt.xticks(m, my_xticks)
plt.plot(m,y0,'ro')
plt.plot(m,y1,'bs')

plt.xlabel('init_mode')
plt.ylabel('Accuracy')

legend1 = plt.legend(['test_score', 'train_score'], loc='lower left')
plt.show()