from baseband import mark5b
import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time
from pulsar.predictor import Polyco



with open('starttimes.txt','r') as f:
	start_times = f.read()
	start_times = start_times.split('\n')[:-1]
	
filename = 'close_pairs.txt'
with open(filename,'r') as f:
	GP_times = f.read()
	GP_times = GP_times.split('\n')[:-1]
	GP_times_utc1 = [entry.split()[0] for entry in GP_times]
	GP_times_utc2 = [entry.split()[1] for entry in GP_times]
	GP_times_unix1 = np.array([Time(t).unix-Time(start_times[0]).unix for t in GP_times_utc1])
	GP_times_unix2 = np.array([Time(t).unix-Time(start_times[0]).unix for t in GP_times_utc2])
	scan_nos1 = [entry.split()[2] for entry in GP_times]
	scan_nos2 = [entry.split()[3] for entry in GP_times]
	#S_Ns = [entry.split()[2] for entry in GP_times]
	
'''	
with open('close_pairs.txt','w') as f:
   for b in close_GP_pairs:
		f.write('{} {} {} {} {}'.format(b[0],b[1],b[2],b[3],b[4])

#Getting close pairs from time strings 

close_GP_pairs = np.zeros(shape = (0,7))
#def get_time_lag():	
for i,ts1 in enumerate(GP_times_utc):
	for j in range(len(GP_times_utc)-i):
		ts2 = GP_times_utc[i+j]
		t_gp1 = Time(ts1)
		t_gp2 = Time(ts2)
		dt = t_gp2-t_gp1
		if abs(dt.sec)<10 and abs(dt.sec)>0.01:
			close_GP_pairs = np.append(close_GP_pairs,np.array([[ts1,ts2,scan_nos[i],scan_nos[i+j],dt.sec]]),0)
			print i,j
			
print close_GP_pairs.shape 
'''


def find_phase(t_GPs,scan_no):
	t0 = Time(start_times[scan_no-1])
	t_GP = Time(t_GPs)
	psr_polyco = Polyco('polyco_new.dat')
	phase_pol = psr_polyco.phasepol(t0)
	phase = np.remainder(phase_pol(t_GP.mjd), 1)
	return phase

phases1 = []
phases2 = []
#def plot_phase_time():
for i,t1 in enumerate(GP_times_utc1):
	phases1.append(find_phase(t1,int(scan_nos1[i])))
	phases2.append(find_phase(GP_times_utc2[i],int(scan_nos2[i])))
phases1 = [phase1%0.8 if phase1>=0.8 else phase1+0.2 for phase1 in phases1]
phases2 = [phase2%0.8 if phase2>=0.8 else phase2+0.2 for phase2 in phases2]
phases1 = np.array(phases1)
phases2 = np.array(phases2)
phases1 = (phases1-1.0/30010*GP_times_unix1)%1.0
phases2 = (phases2-1.0/30010*GP_times_unix2)%1.0	
plt.figure()
plt.plot(phases1,GP_times_unix1,'o')
plt.plot(phases2,GP_times_unix2,'x')
plt.show()

#plot_phase_time()

'''
# The function below is for adding scan numbers to a txt file with times but no scan numbers 

def write_scan_no():
	with open('starttimes.txt','r') as f:
		times = f.read()
		times = times.split('\n')[:-1]
		print len(times)
	text_name = 'all7_sorted.txt'
	with open(text_name, 'r') as f:
		text = f.read()
		text_lines = text.split('\n')[:-1]
		print len(text_lines)
	
	scan_nos = ['0']*len(text_lines)
	for line_no,line in enumerate(text_lines):	
		strings = line.split()
		t_GP = Time(strings[0])
		for i,start_time in enumerate(times):
			ts = Time(start_time)
			ts_next = Time(times[i+1])
			if t_GP.mjd >ts.mjd and t_GP.mjd <ts_next.mjd:
				scan_nos[line_no] = format(i+1,'02d')
				print i
				break
	print scan_nos
	
with open('all7_sorted1.txt','w') as f:
	for i in range(len(text_lines)):
		f.write('{} {}\n'.format(scan_nos[i],text_lines[i]))
'''
