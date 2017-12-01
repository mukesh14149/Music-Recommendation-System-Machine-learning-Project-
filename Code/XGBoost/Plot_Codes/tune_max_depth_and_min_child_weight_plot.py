import matplotlib.pyplot as plt
from matplotlib import pyplot

min_child = [1,5,9,13,17]
epoch = [10,50,100]
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.682419,0.682470,0.682477,0.682451,0.682434]
y1 = [0.714228,0.713973,0.714281,0.714039,0.713302]
y2 = [0.757733 ,0.756241,0.754788 ,0.754389,0.753405]
y3 = [0.792412 ,0.789193,0.783127,0.783127,0.781381]


z0 = [0.682946,0.682996,0.682991,0.682960,0.682941]
z1 = [0.719921,0.719306,0.719444,0.719029,0.718083]
z2 = [0.789645,0.782811,0.778358,0.776173,0.773472]
z3 = [0.883507,0.857825,0.843596,0.833576,0.826753]

plt.plot(min_child,y0,color= colors[0])
plt.plot(min_child,y1,color= colors[1])
plt.plot(min_child,y2,color= colors[2])
plt.plot(min_child,y3,color= colors[3])

plt.plot(min_child,z0,color= colors[0],linestyle='dashed',dashes=(5, 15))
plt.plot(min_child,z1,color= colors[1],linestyle='dashed',dashes=(5, 15))
plt.plot(min_child,z2,color= colors[2],linestyle='dashed',dashes=(5, 15))
plt.plot(min_child,z3,color= colors[3],linestyle='dashed',dashes=(5, 15))


plt.xlabel('min_child_weight')
plt.ylabel('Accuracy')

legend1 = plt.legend(['max_depth = 4','max_depth = 8','max_depth = 12','max_depth= 16'], loc=0)
ax = plt.gca().add_artist(legend1)
legend2 = plt.legend(['train_score = Dashed_line', 'test_score = Solid_line'],loc='upper left')
legend2.legendHandles[0].set_color('black')
legend2.legendHandles[1].set_color('black')

#plt.show()
plt.savefig("tune_max_depth_and_min_child_weight_plot.png")

