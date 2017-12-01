import matplotlib.pyplot as plt
from matplotlib import pyplot

max_depth = [8,10,12,14,16]
x1 = [16,64,256]
x2 = [16,64,256,1024]
x3 = [16,64,256,1024,4096,16384]
x4 = [16,64,256,1024,4096,16384]
x5 = [16,64,256,1024,4096,16384]
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.650006,0.667708,0.682070]
y1 = [0.650536,0.669116,0.691206,0.702824]
y2 = [0.682070,0.669881,0.694292,0.713680,0.716386,0.716386]
y3 = [0.650584,0.670474,0.695542,0.715460,0.723264,0.723605]
y4 = [0.649999,0.670618,0.696291,0.717764,0.724954,0.724889]

z0 = [0.650787,0.671302,0.692862]
z1 = [0.651208,0.672797,0.705710,0.732284]
z2 = [0.651161,0.673635,0.709293,0.762372,0.778986,0.778986]
z3 = [0.651365,0.674199,0.710706,0.767545,0.833182,0.835063]
z4 = [0.650904,0.674296,0.711714,0.771696,0.866219,0.889964]

plt.plot(x1,y0,color= colors[0])
plt.plot(x2,y1,color= colors[1])
plt.plot(x3,y2,color= colors[2])
plt.plot(x4,y3,color= colors[3])
plt.plot(x5,y4,color= colors[4])

plt.plot(x1,z0,color= colors[0],linestyle='dashed',dashes=(5, 15))
plt.plot(x2,z1,color= colors[1],linestyle='dashed',dashes=(5, 15))
plt.plot(x3,z2,color= colors[2],linestyle='dashed',dashes=(5, 15))
plt.plot(x4,z3,color= colors[3],linestyle='dashed',dashes=(5, 15))
plt.plot(x5,z4,color= colors[4],linestyle='dashed',dashes=(5, 15))

plt.xlabel('num_leaves')
plt.ylabel('Accuracy')

legend1 = plt.legend(['max_depth = 8', 'max_depth = 10', 'max_depth = 12', 'max_depth = 14', 'max_depth = 16'], loc='lower right')
ax = plt.gca().add_artist(legend1)
legend2 = plt.legend(['train_score = Dashed_line', 'test_score = Solid_line'])
legend2.legendHandles[0].set_color('black')
legend2.legendHandles[1].set_color('black')

plt.show()
