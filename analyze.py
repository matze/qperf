import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_data(filename):
    data = np.loadtxt(filename)
    w0 = data[data[:,0] == 0]
    w1 = data[data[:,0] == 1]

    plt.plot(w0[:,1], w0[:,3] / w1[:,3], 'o-', label=filename)

if __name__ == '__main__':
    font = {'family' : 'Droid Sans',
            'size'   : 13 }

    # Nice colors stolen from Huy Nguyen at
    # http://www.huyng.com/posts/sane-color-scheme-for-matplotlib/
    axes = {'color_cycle': ['348ABD', '7A68A6', 'A60628', '467821', 'CF4457',
        '188487', 'E24A33']}

    matplotlib.rc('font', **font)

    try:
        # this does not work with older matplotlib version
        matplotlib.rc('axes', **axes)
    except:
        pass

    for filename in sys.argv[1:]:
        plot_data(filename)

    plt.grid(True)
    plt.ylabel('Performance degradation')
    plt.xlabel('Input width')
    plt.legend(loc=4)
    plt.show()
