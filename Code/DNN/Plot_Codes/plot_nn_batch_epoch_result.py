import matplotlib.pyplot as plt
from matplotlib import pyplot

batch = [10,20,40,60,80,100]
epoch = [10,50,100]
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.671144,0.672684,0.672805,0.671406,0.670988,0.671169]
y1 = [0.674998,0.675592,0.675371,0.674608,0.676157,0.675366]
y2 = [0.675327,0.676255,0.676166,0.676672,0.676502,0.676040]

z0 = [0.675820,0.677188,0.676815,0.675347,0.675276,0.675388]
z1 = [0.679870,0.681006,0.680254,0.679501,0.680701,0.680032]
z2 = [0.679844,0.681411,0.681514,0.682217,0.681437,0.681287]

plt.plot(batch,y0,color= colors[0])
plt.plot(batch,y1,color= colors[1])
plt.plot(batch,y2,color= colors[2])

plt.plot(batch,z0,color= colors[0],linestyle='dashed',dashes=(5, 15))
plt.plot(batch,z1,color= colors[1],linestyle='dashed',dashes=(5, 15))
plt.plot(batch,z2,color= colors[2],linestyle='dashed',dashes=(5, 15))


plt.xlabel('batch size')
plt.ylabel('Accuracy')

legend1 = plt.legend(['epoch = 10','epoch = 50','epoch = 100'], loc='lower right')
ax = plt.gca().add_artist(legend1)
legend2 = plt.legend(['train_score = Dashed_line', 'test_score = Solid_line'])
legend2.legendHandles[0].set_color('black')
legend2.legendHandles[1].set_color('black')

plt.show()