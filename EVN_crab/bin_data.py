from baseband import mark5b
import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time

text_name = 'all7_sorted1.txt'
nchan = 32
ctvalues = np.load('./figures/correlation_coeff/{}_{}ctvalues_jun1.npy'.format(text_name,nchan))

bin_no = 100
time_values = ctvalues[:,3]
c3values = ctvalues[:,2]
bins = np.logspace(-1.3,2.5,bin_no)
bin_means = np.histogram(time_values,bins,weights = c3values,density = False)[0]/np.histogram(time_values,bins,density = False)[0]

plt.figure()
plt.semilogx(bins[:-1],bin_means,'-o',label = 'correlation coefficients mean in each of the {} time bin'.format(bin_no))
plt.semilogx(time_values,c3values,'x',label = 'correlation coefficients of pulse pairs')
plt.xlim(0.01,300)
plt.ylim(-0.2,0.3)
plt.legend()
plt.title('Plot of correlation coeffients vs time lag of giant pulse pairs. {} frequency channels per band'.format(nchan))
plt.xlabel('Time lag (s)')
plt.ylabel('Correlation coefficients')
plt.show()
