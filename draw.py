import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(filename, shuffle=True,split_ratio=0.8):
    # Load Data
    df = pd.read_csv(filename,sep=',',header=0)
    # df['x1_2' ] = df['x1'] ** 2
    # df['x2_2' ] = df['x2'] ** 2
    # df['x1x2' ] = df['x1'] * df['x2']
    # df['sinx1'] = np.sin(df['x1'])
    # df['sinx2'] = np.sin(df['x2'])
    labels = df['y'].values
    del df['y']
    points = df.values

    # Shuffle data
    if shuffle == True:
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        points = np.array(points)[indices]
        labels = np.array(labels)[indices]

    # Divide the data
    N = int(split_ratio * len(labels))

    return points[:N], labels[:N], points[N:], labels[N:]

data_path  = "data/data1_4.csv"

X_train, Y_train, X_test, Y_test = load_data(data_path,True,0.8)
points = X_train
labels = Y_train

fig= plt.figure()
ax = fig.add_axes([0.1,0.1,0.85,0.85])
ax.grid(color='b', ls = '-.', lw = 0.25)
ax.set_xlabel("x axis")
ax.set_ylabel("y axis")
x_lim = np.max(points[:,0]) + 2
y_lim = np.max(points[:,1]) + 2
ax.set_xlim(-x_lim, x_lim)
ax.set_ylim(-y_lim, y_lim)

d = np.arange(-x_lim,x_lim)
ax.plot(d,d*0)  # Draw y-axis
ax.plot(d*0,d)  # Draw x-axis

# Draw Data
b_x = []; b_y = []
r_x = []; r_y = []
for i in range(len(points)):
    x = points[:,0][i]
    y = points[:,1][i]
    if labels[i] == 1:
        r_x.append(x)
        r_y.append(y)
    else:
        b_x.append(x)
        b_y.append(y)
ax.plot(b_x,b_y, 'b.')
ax.plot(r_x,r_y, 'r.')
plt.show()
