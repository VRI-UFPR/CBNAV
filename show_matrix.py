import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import time

#fig, (ax1, ax2) = plt.subplots(2)

while True:
    reward_matrix = np.load('new_results/reward_matrix_6.npy', allow_pickle=True)
    #reward_matrix = np.load('new_results/conv_reward_matrix_498.npy', allow_pickle=True)
    print(reward_matrix[98, 66])
    plt.xlabel('Enviroment X (m)')
    plt.ylabel('Enviroment Y (m)')
    plt.tight_layout()
    plt.imshow(reward_matrix.T, cmap=plt.get_cmap('YlOrRd') , origin='lower', vmin='-1', vmax='6', extent = [0 , 21, 0 , 21])
    plt.pause(0.01)
    #plt.show()
    time.sleep(2)

'''
test_num = 6
fout = open('new_log/all_'+str(test_num), "r")

r1 = []
r3 = []
reward = []
reward_matrix = []
path_matrix = []
count =[]

_count = 0
for line in fout:
    _r1 = float(line.split()[0])
    r1.append(_r1)
    
    _r3 = float(line.split()[1])
    r3.append(_r3)

    _reward = float(line.split()[2])
    reward.append(_reward)

    _reward_matrix = float(line.split()[3])
    reward_matrix.append(_reward_matrix)

    _path_matrix = float(line.split()[4])
    path_matrix.append(_path_matrix)

    count.append(_count)

    _count += 1

    if (_count == 1000):
        break

fout.close 

plt.scatter(r1, reward)
plt.show()
'''

#new = gaussian_filter(reward_matrix, sigma=5)
'''
new = np.minimum(reward_matrix, 2)

for x in range(0, reward_matrix.shape[0]):
    for y in range(0, reward_matrix.shape[1]):
        if reward_matrix[x, y] == 1:
            reward_matrix[x, y] = 0

new = gaussian_filter(reward_matrix, sigma=8)
'''

'''
def visualize_log(filename, figsize=None, output=None):
    with open(filename, 'r') as f:
        data = json.load(f)
    if 'episode' not in data:
        raise ValueError('Log file "{}" does not contain the "episode" key.'.format(filename))
    episodes = data['episode']

    # Get value keys. The x axis is shared and is the number of episodes.
    keys = sorted(list(set(data.keys()).difference(set(['episode']))))

    if figsize is None:
        figsize = (15., 5. * len(keys))
    f, axarr = plt.subplots(len(keys), sharex=True, figsize=figsize)
    for idx, key in enumerate(keys):
        axarr[idx].plot(episodes, data[key])
        axarr[idx].set_ylabel(key)
    plt.xlabel('episodes')
    #plt.tight_layout()
    if output is None:
        plt.show()
    else:
        plt.savefig(output)


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='The filename of the JSON log generated during training.')
parser.add_argument('--output', type=str, default=None, help='The output file. If not specified, the log will only be displayed.')
parser.add_argument('--figsize', nargs=2, type=float, default=None, help='The size of the figure in `width height` format specified in points.')
args = parser.parse_args()

# You can use visualize_log to easily view the stats that were recorded during training. Simply
# provide the filename of the `FileLogger` that was used in `FileLogger`.
visualize_log(args.filename, output=args.output, figsize=args.figsize)

'''