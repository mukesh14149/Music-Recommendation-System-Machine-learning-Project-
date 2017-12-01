import matplotlib.pyplot as plt
from matplotlib import pyplot

x = [1,10,20,30,40,50,60,70,80,90,100]
colors=["#FF0000","#008080","#FF00FF","#808000", "#800080","#00FFFF","#0000FF","#800080","#FF00FF"]

y0 = [0.595061,0.596846,0.596238,0.601011,0.603300,0.602781,0.599904,0.595914,0.594776,0.594310,0.592277]
y1 = [0.600878,0.601934,0.601294,0.604662,0.606211,0.605651,0.603433,0.599541,0.598235,0.597827,0.598646]
plt.plot(x,y0,'ro')
plt.plot(x,y1,'bs')

plt.xlabel('neurons')
plt.ylabel('Accuracy')

legend1 = plt.legend(['test_score', 'train_score'], loc='lower left')
plt.show()