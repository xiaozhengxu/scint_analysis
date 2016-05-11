from GPbaseband import *
from baseband import mark5b
import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time


text_name = 'all20.txt'
with open(text_name, 'r') as f:
	text = f.read()
	text_lines = text.split('\n')

print len(text_lines)
tvalues = []
cvalues = []

for delta in [1]:	
	for i in range(len(text_lines)-delta):
		j = i+delta	

		strings1 = text_lines[i].split()
		strings2 = text_lines[j].split()
		if text_name[0] == 'a': # get information from a all__.txt file
			scan_no1 = strings1[1]
			scan_no2 = strings2[1]
			t1 = strings1[2]
			t2 = strings2[2]
		elif text_name[0] == 's': # get information from a scan__.txt file
			scan_no1 = text_name[4:6]
			scan_no2 = text_name[4:6]
			t1 = strings1[0]
			t2 = strings2[0]
			
		fn1 = '/cita/h/home-2/xzxu/trails/data/ek036a_ef_no00{}.m5a'.format(scan_no1)
		fn2 = '/cita/h/home-2/xzxu/trails/data/ek036a_ef_no00{}.m5a'.format(scan_no2)
		t_gp1 = Time(t1)
		t_gp2 = Time(t2)
		dt = t_gp2-t_gp1
		if abs(dt.sec)<600 and abs(dt.sec)>10e-5:
			try:	
				gp1 = GP_data(fn1,t_gp1)
				gp2 = GP_data(fn2,t_gp2)
				c = get_correlation_coefficients(gp1,gp2)
				tvalues.append(abs(dt.sec))
				cvalues.append(c[0])
				print scan_no1,t1,scan_no2,t2
				print 'The correlation coefficient is {}, the time lag between the two pulses is {} seconds.'.format(c[0],dt.sec)
			except AssertionError:
				continue

plt.figure()
plt.semilogx(tvalues,cvalues,'bo')
plt.xlabel('time lag (seconds)')
plt.ylabel('Correlation Coefficient')
plt.show()
