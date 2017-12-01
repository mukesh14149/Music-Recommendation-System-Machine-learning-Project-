import matplotlib.pyplot as plt
from matplotlib import pyplot

momentum = [0.0,0.2,0.4,0.6,0.8,0.9]
m = [0.001,0.01,0.1,0.2,0.3]
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.673283,0.674372,0.665141,0.659343,0.656233]
y1 = [0.673316,0.673980,0.665510,0.656796,0.647963]
y2 = [0.674083,0.674276,0.664777,0.650534,0.639731]
y3 = [0.674605,0.672824,0.660082,0.637437,0.606578]
y4 = [0.674638,0.668987,0.640021,0.500041,0.499988]
y5 = [0.674316,0.664970,0.499980,0.499995,0.500271]
z0 = [0.677255,0.679808,0.670855,0.665140,0.661224]
z1 = [0.677713,0.679170,0.670484,0.661414,0.652098]
z2 = [0.678322,0.679509,0.669484,0.655434,0.643506]
z3 = [0.679288,0.678216,0.664363,0.641958,0.608726]
z4 = [0.679816,0.675017,0.644205,0.500151,0.500000]
z5 = [0.679448,0.670643,0.500011,0.500000,0.500643]

plt.plot(m,y0,color= colors[0])
plt.plot(m,y1,color= colors[1])
plt.plot(m,y2,color= colors[2])
plt.plot(m,y3,color= colors[3])
plt.plot(m,y4,color= colors[4])
plt.plot(m,y5,color= colors[5])

plt.plot(m,z0,color= colors[0],linestyle='dashed',dashes=(5, 15))
plt.plot(m,z1,color= colors[1],linestyle='dashed',dashes=(5, 15))
plt.plot(m,z2,color= colors[2],linestyle='dashed',dashes=(5, 15))
plt.plot(m,z3,color= colors[3],linestyle='dashed',dashes=(5, 15))
plt.plot(m,z4,color= colors[4],linestyle='dashed',dashes=(5, 15))
plt.plot(m,z5,color= colors[5],linestyle='dashed',dashes=(5, 15))


plt.xlabel('learning rate')
plt.ylabel('Accuracy')

legend1 = plt.legend(['m = 0.0', 'm = 0.2', 'm = 0.4', 'm = 0.6', 'm = 0.8', 'm = 0.9'], loc='lower left')
ax = plt.gca().add_artist(legend1)
legend2 = plt.legend(['train_score = Dashed_line', 'test_score = Solid_line'])
legend2.legendHandles[0].set_color('black')
legend2.legendHandles[1].set_color('black')

plt.show()