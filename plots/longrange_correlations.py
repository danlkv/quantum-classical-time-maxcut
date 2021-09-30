import matplotlib.pyplot as plt
from matplotlib.cm import gnuplot2
import numpy as np
import json

def main():
    ddout = json.load(open('../data/longrange_correlations.json'))
    N = 6
    colors = iter([gnuplot2(x) for x in np.linspace(0.2, 0.8, N-1)])
    plt.figure(figsize=(8,6))
    #offset = 0.2
    #colors = iter([gnuplot2(offset + x*(1-2*offset)) for x in np.linspace(0, 1, N)])

    for p in range(1,N):
        p = str(p)
        dist = np.arange(1,len(ddout[p])+1)
        plt.semilogy(dist[0:-1],
                     -np.array(ddout[p][0:-1])*(-1)**(dist[0:-1]+1),
                     '.-',
                     label='p='+str(p),
                     color=next(colors)
                    )
    plt.legend()
    plt.xticks(np.arange(1,11))
    plt.xlabel('Distance between vertices')
    plt.ylabel('Correlation $(-1)^{d}\\langle \\sigma_i\\sigma_{i+d}\\rangle$')
    plt.savefig('./pdf/longrange_correlations.pdf')

if __name__=='__main__':
    main()
    plt.show()
