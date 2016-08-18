
# coding: utf-8

# Processing the frequency spectrum of a pulse into time lag series

# In[1]:

import numpy as np
from scipy.stats.stats import pearsonr
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import ICRS, Galactic, FK4, FK5 

from os.path import exists 

#Using latex rendering:
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


# In[3]:

get_ipython().magic(u'pylab inline')
colors = [(31, 119, 180),(255, 127, 14), (44, 160, 44),(148, 103, 189),(214, 39, 40)] #blue, orange, green, purple, red
for i in range(len(colors)):    
    r, g, b = colors[i]    
    colors[i] = (r / 255., g / 255., b / 255.) 
    


# Loading a giant pulse 

# In[ ]:

fs1.shape


# In[3]:

i = 235
nchan = 32
tele1 = 'jb'
tele2 = 'ef'
time_string1 = '2015-10-19T00:17:47.415' # S/N = 89!, main pulse 
time_string2 = '2015-10-19T02:35:26.143' # S/N = 43, main pulse 
time_string3 = '2015-10-19T02:06:52.280' #S/N = 7.9, inter pulse
time_string4 = '2015-10-19T02:06:58.031' #S/N = 9.6, main pulse
time_string5 = '2015-10-19T00:55:15.557' #S/N = 8.33, main pulse
time_string6 = '2015-10-19T00:55:49.673' #S/N = 8.55, main pulse
time_string7 = '2015-10-19T01:29:52.509' #S/N = 8.44, main pulse
time_string8 = '2015-10-19T01:55:06.340' #S/N = 8.66, main pulse
time_string9 = '2015-10-19T02:13:46.551' #S/N = 9.56, main pulse
time_string10 = '2015-10-19T02:36:11.418'#S/N = 8.41, main pulse

time_string11 = '2015-10-19T00:54:59.880' # bright double pulse
time_string12 = '2015-10-19T02:36:46.815'  #bright double pulse2

time_string13 = '2015-10-19T00:37:14.418' # close main pulse pair
time_string14 = '2015-10-19T00:37:15.126'

time_string15 = '2015-10-19T01:57:05.107' # the closest main pulse pair 
time_string16 = '2015-10-19T01:57:05.410'

time_string17 = '2015-10-19T00:56:55.579' # close pair by 1.078s
time_string18 = '2015-10-19T00:56:54.501' 

time_string19 = '2015-10-19T02:17:06.664' #close pair by 1.281s
time_string20 = '2015-10-19T02:17:07.945' 

time_string21 = '2015-10-19T01:28:04.429' #close pair by 1.65
time_string22 = '2015-10-19T01:28:06.081' 

time_string23 = '2015-10-19T01:44:15.701' # close pair by 1.98
time_string24 = '2015-10-19T01:44:17.690' 

time_string25 = '2015-10-19T01:29:52.509' # close main pulse pair by 2.09
time_string26 = '2015-10-19T01:29:54.600' 

time_string27 = '2015-10-18T23:47:20.315' # close interpulse and main pulse pair (interpulse) by 0.49s
time_string28 = '2015-10-18T23:47:20.807' #main pulse for the above pair 


# In[14]:

def get_cc((fs1,bgstd1),(fs2,bgstd2),N):
    mean1 = np.mean(fs1)
    mean2 = np.mean(fs2)
    numerator = np.mean((fs1-mean1)*(fs2-mean2))
#     denominator = np.sqrt((np.std(fs1)**2-(bgstd1/np.sqrt(N))**2)*(np.std(fs2)**2-(bgstd2/np.sqrt(N))**2))
    denominator = np.std(fs1)*np.std(fs2)
    return numerator/denominator
    
def get_ccs((fs1,bgstd1),(fs2,bgstd2),N):
    coefficients= []
    for i in range(8):
        X = fs1[i]
        Y = fs2[i]
        Xbg = bgstd1[i]
        Ybg = bgstd2[i]
        coefficient = get_cc((X,Xbg),(Y,Ybg),N)
        coefficients.append(coefficient) 
    return np.mean(coefficients),coefficients


# In[5]:

def get_SN(outputsumfreq):
    noise_std = np.std(outputsumfreq[:20]) 
    noise_mean = np.mean(outputsumfreq[:20])
    sigs_noise = (outputsumfreq-noise_mean)/noise_std
#     print 'standard deviation of noise is ', noise_std, 'the mean noise is ', noise_mean
    return sigs_noise

def process_freq_spec(gp1,nchan,index=None,draw=0):
    '''function takes a continuous frequency spectrum and cuts it into 8 bands, divide by the giant pulse mean, and normalize it by itself gp1=freq_spec[i]. function also plots the frequency spectrum'''
    gp1_8=np.array([gp1[int(50./512*nchan):int(460./512*nchan)]])
    for j in range(7):gp1_8=np.append(gp1_8,np.array([gp1[(j+1)*(nchan+1)+int(50./512*nchan):(j+1)*(nchan+1)+int(460./512*nchan)]]),0)
#     gp1_8/=gpab #Dividing by giant pulse average
#     for i in range(8):gp1_8[i]=gp1_8[i]/gp1_8[i].mean()-1.
    if nchan == 512:
        new1 = np.zeros((8,25))
        for i in range(8):
            for j in range(25):
                new1[i,j] = average(gp1_8[i,j*N:(j+1)*N])
        return new1,gp1_8
    return gp1_8

def load_gp(time_string,telescope,nchan,draw = 0):
    '''Takes in a time string, a string indicating whcih telescope to use, and an interger of nchan'''
    gp1 = np.load('/mnt/raid-cita/ramain/GPs/GP{}/p{}.npy'.format(telescope,time_string))
    gp1t = np.fft.irfft(gp1,axis=1)
    gp1s = gp1t.reshape(-1,2*nchan,16)
    gp1s = np.fft.rfft(gp1s,axis=1)
    dchan = gp1s
    dR = np.concatenate((dchan[:,::-1,8], dchan[...,0], dchan[:,::-1,10], dchan[...,2], dchan[:,::-1,12], dchan[...,4], dchan[:,::-1,14], dchan[...,6]), axis=1)
    dL = np.concatenate((dchan[:,::-1,9], dchan[...,1], dchan[:,::-1,11], dchan[...,3], dchan[:,::-1,13], dchan[...,5], dchan[:,::-1,15], dchan[...,7]), axis=1)
    output = (abs(dR)**2 + abs(dL)**2).T

    outputsumfreq = output.sum(0) #dedispersed
    #computes signal to noise after summing all the frequencies
    sigs_noise = get_SN(outputsumfreq)
    sn_sorted = np.argsort(sigs_noise)

    peak_time = np.argmax(sigs_noise)
    S_N = outputsumfreq[peak_time]/np.mean(outputsumfreq[:30])
    
    if nchan ==512:
        background_std = []
        for j in range(8):
            background_std.append(output[j*(nchan+1)+40./512*nchan:j*(nchan+1)+465./512*nchan,:peak_time-3].std())
        output_pulse = output[:,peak_time]
#         print 'std2 =',background_std2
        background_freq = output[:,peak_time-3]
        
    if nchan == 128:
        sn_sorted = np.argsort(sigs_noise)
#         print sn_sorted
#         print sn_sorted.shape
        background_freq = (output[:,sn_sorted[30]]+output[:,sn_sorted[31]])/2
        output_pulse = (output[:,sn_sorted[-1]]+output[:,sn_sorted[-2]])/2
        background_std = []
        for j in range(8):
            background_std.append(background_freq[j*(nchan+1)+40./512*nchan:j*(nchan+1)+465./512*nchan].std())

    if nchan == 32:
#         print sn_sorted
        background_1= np.concatenate([np.array([output[:,sn_sorted[-k]]]) for k in range(51,53)],axis = 0).mean(0) 
        output_pulse = np.concatenate([np.array([output[:,sn_sorted[-k]]]) for k in range(1,3)],axis = 0).mean(0)
        background_std = []
        for j in range(8):
            background_std.append(background_1[j*(nchan+1)+40./512*nchan:j*(nchan+1)+465./512*nchan].std())
        background_freq = background_1
    freq_spec = output_pulse-background_freq
    
    if nchan == 16:
        sn_sorted = np.argsort(sigs_noise)
        background_1 = output[:,sn_sorted[-101]]
        output_pulse = output[:,sn_sorted[-1]]
        background_std = []
        for j in range(8):
            background_std.append(background_1[j*(nchan+1)+40./512*nchan:j*(nchan+1)+465./512*nchan].std())
        background_freq = background_1

    if draw:
        figure(figsize = (10,3))
        plot(outputsumfreq)
        figure()
        plot(output_pulse)
        plot(background_freq)
        figure()
        plot(freq_spec)
        
    return freq_spec,background_std,sigs_noise,outputsumfreq
    

nchan = 32
S_N3= 4
S_N4 =4
# time_string = text_lines[22]
time_string_1 = time_string1
time_string_2 = time_string5


fs1,bgstd1,os1,sn1 = load_gp(time_string_1,'jb',nchan,draw = 0) 
fs2,bgstd2,os2,sn2 = load_gp(time_string_1,'ef',nchan,draw = 0)


# fgp1 = process_freq_spec(fs1,nchan,i-1,draw = 0)
# fgp2 = process_freq_spec(fs2,nchan,i-1, draw = 0)



# # plot_2_freq(fgp1,fgp2,[i-1,i-1])
# # plot_2_timelag(tgp1,tgp2,[i-1,i-1])

figure(figsize = (10,7))
plot(os1,label = 'jb')
plot(os2,label = 'ef')
ylabel('signal to noise')
xlabel(r'time ($\mu$s)')
legend()



# In[6]:

teles = ['jb','ef','hh','wb','o8','tr','bd','sr']

f,axarr=plt.subplots(8,1,figsize=(10,15))
for i in range(8):
    fs,_,_,_ = load_gp(time_string1,teles[i],128,draw = 0)
    axarr[i].plot(fs,label = teles[i])
    axarr[i].legend()
axarr[0].set_title('The telescope gain shapes of six telescopes')



# In[8]:

figure(figsize = (10,7))
for i in range(8):
    fs,_,sn,ops = load_gp(time_string1,teles[i],128,draw = 0)
    plot(sn,label = teles[i])
legend()
title('Signal to noise (devided by noise sigmas) of pulse')
xlabel('time',fontsize = 12)
ylabel('S/N',fontsize = 12)


# Getting telescope values

# In[39]:

'''Getting correlation coefficient vs telescope distance plot'''
teles = ['jb','ef','hh','wb','o8','tr','sr','bd']
N = 16
D = {}
nchan = 512
for i,tele1 in enumerate(teles):
    for j,tele2 in enumerate(teles[i+1:]):
        fs1,bgstd1,os1,sn1 = load_gp(time_string1,tele1,nchan,draw = 0) 
        fs2,bgstd2,os2,sn2 = load_gp(time_string1,tele2,nchan,draw = 0)
        fgp1,fgp1l = process_freq_spec(fs1,nchan,i-1,draw = 0)
        fgp2,fgp2l = process_freq_spec(fs2,nchan,i-1, draw = 0)
        cc,ccs = get_ccs((fgp1,bgstd1),(fgp2,bgstd2), N)
        D[tele1+'-'+tele2] = (find_distance(obspos[tele1],obspos[tele2]),cc)
print D


# In[23]:

distances,cvalues = zip(*D.values())
figure(figsize = (10,7))
plot(distances,cvalues,'*')
xlabel('Distance (km)',fontsize = 12)
ylabel('Correlation coefficient',fontsize = 12)
for i in range(len(circle_sizescc)):
    annotate(D[coercoeff[i]],zip(EWs,NSs)[i],xytext = (-10, 10),textcoords = 'offset points',fontsize = 12)
title('Correlation coefficient of the brightest pulse at 2 telescopes vs the distance of the 2 telescopes',fontsize = 12)


# In[10]:

def find_distance(pos1,pos2):
    '''Returns the distance of 2 positions in km'''
    return sqrt((pos1.x-pos2.x)**2+(pos1.y-pos2.y)**2+(pos1.z-pos2.z)**2).to(u.km).value
find_distance(obspos['ef'],obspos['hh'])


# In[16]:

def find_EW_NS_positions(pos1,pos2):
    '''Returns the distance along EW and along NS as a tuple value'''
    R = 6371 #earth's radius in km
    dtheta = pos1.longitude.value-pos2.longitude.value #in degrees
    phi = (90-abs(pos1.latitude.value)+90-abs(pos2.latitude.value))/2.
    EW = abs(2.*R*sin(deg2rad(dtheta)/2.)*sin(deg2rad(phi)))
    NS = abs(pos1.z.value-pos2.z.value)/1000.
    return EW,NS
find_EW_NS_positions(obspos['ef'],obspos['sr'])


# In[29]:

teles = ['jb','ef','hh','wb','o8','tr','sr','bd']
N = 16
D = {}
EWs = []
NSs = []
coercoeff = []
nchan = 512
for i,tele1 in enumerate(teles):
    for j,tele2 in enumerate(teles[i+1:]):
        fs1,bgstd1,os1,sn1 = load_gp(time_string1,tele1,nchan,draw = 0) 
        fs2,bgstd2,os2,sn2 = load_gp(time_string1,tele2,nchan,draw = 0)
        fgp1,fgp1l = process_freq_spec(fs1,nchan,i-1,draw = 0)
        fgp2,fgp2l = process_freq_spec(fs2,nchan,i-1, draw = 0)
        cc,ccs = get_ccs((fgp1,bgstd1),(fgp2,bgstd2), N)
        EWs.append(find_EW_NS_positions(obspos[tele1],obspos[tele2])[0])
        NSs.append(find_EW_NS_positions(obspos[tele1],obspos[tele2])[1])
        coercoeff.append(cc)
        D[cc] = (tele1+'-'+tele2)
# print EWs
# print NSs
print D


# In[38]:

figure(figsize = (12,8))
circle_sizescc = [cir**4*400. for cir in coercoeff]
scatter(EWs,NSs,s=circle_sizescc,alpha = 0.5)
xlabel('Distance along the East west axis (km)',fontsize = 18)
ylabel('Distance along the North south axis (km)',fontsize = 18)
tick_params(axis='both', which='major', labelsize=18)
for i in range(len(circle_sizescc)):
    annotate(D[coercoeff[i]],zip(EWs,NSs)[i],xytext = (-10, 10),textcoords = 'offset points',fontsize = 12)
title('Correlation coefficient of the brightest pulse at different telescopes. \nThe circle size is proportional the correlation coefficient to the power of four.',
     fontsize = 14)
show()


# In[45]:

obspos['ef'].z-obspos['hh'].z


# In[11]:

obspos = {'ef': EarthLocation(4033947.2616*u.m, 486990.7866*u.m, 4900430.9915*u.m),
         'wb': EarthLocation(3828763.6100*u.m, 442448.8200*u.m, 5064923.0800*u.m),
         'jb': EarthLocation(3822626.0400*u.m, -154105.6500*u.m, 5086486.0400*u.m),
         'sr': EarthLocation(4864197.4692*u.m, 792184.8623*u.m, 4035367.1273*u.m),
         'tr': EarthLocation(3638558.5100*u.m, 1221969.7200*u.m, 5077036.7600*u.m),
         'o8': EarthLocation(3370965.9090*u.m, 711466.1978*u.m, 5349664.1947*u.m),
         'hh': EarthLocation(5083275.3295*u.m, 2668779.2123*u.m, -2769143.9034*u.m),
         'bd': EarthLocation(-839566.7739*u.m, -3866810.1085*u.m, 4985708.7210*u.m)
         }
efp = obspos['ef']


# In[20]:

efp.geodetic


# In[40]:

print obspos['hh'].geodetic
print deg2rad(obspos['ef'].latitude.value)
print deg2rad(obspos['hh'].latitude.value)



# In[14]:



cc12,ccs12 = get_ccs((new1,bgstd1),(new2,bgstd2), N = N)
# cc12,ccs12 = 0.,numpy.zeros(8)
cc512,ccs512 = get_ccs((fgp1,bgstd1),(fgp2,bgstd2), N = 1)

print '\nthe two pulses are ',cc12,ccs12, 'correlated (25 values)'
print '\nthe two pulses are ',cc512,ccs512, 'correlated ({} channels)'.format(nchan)

t_gp1 = Time(time_string_1)
t_gp2 = Time(time_string_2)
dt = t_gp2-t_gp1

for i in range(8):
    figure(figsize = (10,3))
    plot(linspace(0,410,num_values),new1[i]/np.mean(new1[i])-1.,label = 'pulse1-averaged',color = colors[0])
    plot(linspace(0,410,num_values),new2[i]/np.mean(new2[i])-1.,label = 'pulse2-averaged',color = colors[1])
    plot(fgp1[i]/fgp1[i].mean()-1.,label = 'pulse1',color = colors[0],linewidth = 0.5)
    plot(fgp2[i]/fgp2[i].mean()-1.,label = 'pulse2',color = colors[1],linewidth = 0.5)
    legend()
    title('Correlation of two main pulses separated by {}s. The correlation coefficient of the original 410 frequency values is {}.            \n The correlation coefficient of the averaged 25 values is {}.'.format(round(abs(dt.sec),3),round(ccs12[i],3),round(ccs512[i],3)))
    xlabel('Frequency channel')
    ylabel('Power')


# In[26]:

sn1s = []
sn2s = []
c_cs = []
nchan = 32
for i in range(len(text_lines[:-1])):
    fs1,bgstd1,os1,sn1 = load_gp(text_lines[i],'wb',nchan,draw = 0) 
    fs2,bgstd2,os2,sn2 = load_gp(text_lines[i],'ef',nchan,draw = 0)
    fs3,bgstd3,os3,sn3 = load_gp(text_lines[i],'jb',nchan,draw = 0)
    if sn1>3 or sn2>3 or sn3>3:
        fgp1 = process_freq_spec(fs1,nchan,i-1,draw = 0)
        fgp2 = process_freq_spec(fs2,nchan,i-1, draw = 0)
        fgp3 = process_freq_spec(fs3,nchan,i-1, draw = 0)
        cc12,ccs12 = get_ccs((fgp1,bgstd1),(fgp2,bgstd2))
        cc23,ccs23 = get_ccs((fgp2,bgstd2),(fgp3,bgstd3))
        cc13,ccs13 = get_ccs((fgp1,bgstd1),(fgp3,bgstd3))
#         figure(figsize = (10,3))
#         plot(fgp1[0]/fgp1[0].mean()-1.,label = 'wb',color = colors[0])
#         plot(fgp2[0]/fgp2[0].mean()-1.,label = 'ef',color = colors[1])
#         plot(fgp3[0]/fgp3[0].mean()-1.,label = 'jb',color = colors[2])
#         legend()
#         title('Correlation of a pulse at wb(S/N = {}), ef(S/N = {}) and jb(S/N = {}). \n The correlation coefficient calculated for the 1st band is {}(ef-jb), {}(ef-wb), {}(wb-jb)'.format(round(sn1,3),round(sn2,3),round(sn3,3),round(ccs23[0],3),round(ccs12[0],3),round(ccs13[0],3)))
        c_cs.append(cc12)
        sn1s.append(sn1)
        sn2s.append(sn2)
    


# In[28]:

figure(figsize = (10,3))
plot(sn1s,c_cs,'x')
# plot(sn2s,c_cs,'x')
# xlim(0,30)
xlabel('S/N at jb')
title('Correlation coefficient of the power spectres of the same pulse at jb and ef')
ylabel('Correlation coefficient')
bin_nos = 20
bins = np.linspace(0,50,bin_nos)
ccs_binned = np.histogram(sn2s,bins,weights = c_cs,density = False)[0]/np.histogram(sn2s,bins,density = False)[0]
plot(bins[:-1],ccs_binned)
ccs_binned


# In[ ]:

# cc,ccs=get_ccs((fgp1,bgstd1),(fgp2,bgstd2))
# print '\nthe two pulses are', cc ,'correlated'
# print ccs

# print 'numerator for 1st band:', np.mean((fgp1[0]-fgp1[0].mean())*(fgp2[0]-fgp2[0].mean()))
# print 'denominator for 1st band:', np.sqrt(((np.std(fgp1[0]))**2-bgstd1**2)*((np.std(fgp2[0]))**2-bgstd2**2))

# figure(figsize = (15,3))
# plot(fgp1[0],label = tele1)
# plot(fgp2[0],label = tele2)
# title('A bright pulse at {} on {} and {} telescopes. nchan = {}. cc calculated = {}'.format(time_string,tele1,tele2,nchan,round(ccs[0],4)))
# legend()

# figure(figsize = (15,3))
# plot(fs1bg,label = 'noise-ef')
# plot(fs2bg,label = 'noise-jb')
# title('noise of the pulse at different telescopes')
# legend()


# In[ ]:

'''Frequency shift '''
# cc12,ccs12 = get_ccs((fgp1,fbg1),(fgp2,fbg2))
# # cc23,ccs23 = get_ccs((fgp2,bgstd2),(fgp3,bgstd3))
# # cc13,ccs13 = get_ccs((fgp1,bgstd1),(fgp3,bgstd3))
# print '\nthe two pulses are ',cc12,ccs12, 'correlated'

# cc_dfs = []
# for k in range(fgp1.shape[1]):
#     cc,ccs = get_ccs(((np.roll(fgp1,k,axis =1)),bgstd1),(fgp2,bgstd2))
#     cc_dfs.append(cc)

    
# # f,axarr=plt.subplots(8,1,figsize=(10,15))
# # for i in range(8):
# #     axarr[i].plot(fgp1[i]/fgp1[i].mean()-1.)
# #     axarr[i].plot(fgp2[i]/fgp2[i].mean()-1.)
    
# figure(figsize = (7,5))
# plot(cc_dfs)
# xlabel('frequency shift (frequency units)')
# xlim(-10,len(cc_dfs))
# title('correlation coefficient as a function of freq shift of a pair of pulses separated by {}'.format(round(abs(dt.sec),3)))

print fgp1.shape


# FOr testing getting rid of noise:

# In[5]:

#For testing binning!
# figure()
# binning_constant = np.histogram(np.linspace(0,15,213),np.linspace(0,15,16))[0]
# hist = np.histogram(np.linspace(0,15,213),np.linspace(0,15,16),weights = abs(tgp1[0]))[0]
# # plot(hist)
# # plot(binning_constant)
# plot(np.linspace(0,15,213),abs(tgp1[0]))
# plot(np.linspace(0,15,15),hist/binning_constant)


# In[70]:

def bin_data(t,a,bins):
    '''This function takes in t( an array of time values), a (array of data), and the bins and return an array of the binned data a'''
    a_binned = np.histogram(t,bins,weights = a,density = False)[0]/np.histogram(t,bins,density = False)[0]
    return a_binned 

def transform_to_timelag(fgp,index= None,draw = 0,binned=0):
    if binned ==0:
        N = 213
        tgp=np.zeros(shape=(0,N))
        for i in range(8):tgp=np.append(tgp,np.array([np.fft.rfft(fgp[i])]),0)
    else:
        N = 15
        tgp = np.zeros(shape=(0,N))
        for i in range(8):
            a = np.fft.rfft(fgp[i])
            tgp=np.append(tgp,np.array([bin_data(np.linspace(0,15,213),abs(a),np.linspace(0,15,16))]),0)    
    if draw:
        f,axarr=plt.subplots(8,1,figsize=(10,15))
        for i in range(8):axarr[i].plot(np.linspace(0,15,N),abs(tgp[i]))
        axarr[0].set_title('Time lag spectrum of giant pulse at {}'.format(time_values[index]))
        axarr[7].set_xlabel('Time lag(\mu s)')
    return tgp


# tgp1b = transform_to_timelag(fgp1,draw = 0,binned = 1)
# tgp2b = transform_to_timelag(fgp2,draw = 0,binned = 1)
# tgp1 = transform_to_timelag(fgp1,draw = 0,binned = 0)
# tgp2 = transform_to_timelag(fgp2,draw = 0,binned = 0)
# N=tgp1[0].shape[0]
# f,axarr=plt.subplots(8,1,figsize=(10,15))
# for k in range(8):
#     #normalized,binned:
# #     axarr[k].plot(np.linspace(0,15,N),(abs(tgp1b[k]))/np.mean(abs(tgp1b[k]))-1,label = tele1)
# #     axarr[k].plot(np.linspace(0,15,N),(abs(tgp2b[k]))/np.mean(abs(tgp2b[k]))-1,label = tele2)
#     #original units, binned:
# #     axarr[k].plot(np.linspace(0,15,15),abs(tgp1b[k]),label = tele1+' binned')
# #     axarr[k].plot(np.linspace(0,15,15),abs(tgp2b[k]),label = tele2+' binned')
#     #original units, not binned:
#     axarr[k].plot(np.linspace(0,15,213),abs(tgp1[k]),'b',label = tele1)
#     axarr[k].plot(np.linspace(0,15,213),abs(tgp2[k]),'g',label = tele2)
    
# axarr[0].set_title('Time lag spectrum of giant pulse at {} (S/N ={}) on telescopes {} and {}'.format(time_string4,S_N4,tele1,tele2))
# axarr[7].set_xlabel('Time lag(\mu s)')
# legend(loc=(0.95,0.5))


# In[7]:

text_lines[i-1]


# In[66]:

nchan = 512

def plotmean(tgp10):
    figure(figsize=(10,6))
    plot(np.linspace(0,16,213),abs(tgp10).mean(0))
    title('Mean of time lag spectrum')
    
def find_std(tgp,draw = 0):
    stda=np.std(tgp,axis=0) #numpy's std deals with complex numbers: std = sqrt(mean(abs(x - x.mean())**2)) 
    return stda
    if draw:
        figure(figsize=(10,6))
        plot(np.linspace(0,16,213),stda)
        title('Standard deviation as a function of time lag')

def plot_2_freq(fgp1,fgp2,index):
    '''index is an list of 2 indices'''
    f,axarr=plt.subplots(8,1,figsize=(10,15))
    for i in range(8):
        axarr[i].plot(fgp1[i])
        axarr[i].plot(fgp2[i])
    axarr[0].set_title('Frequency spectrum of giant pulse at {}, {}'.format(time_values[index[0]],time_values[index[1]]))
    axarr[7].set_xlabel('Frequency channels (512 channels per 16MHz = 0.03MHz per channel)')

def plot_2_timelag(tgp1,tgp2,index):
    '''index is an list of 2 indices'''
    f,axarr=plt.subplots(8,1,figsize=(10,15))
    for i in range(8):
        axarr[i].plot(np.linspace(0,15,213),abs(tgp1[i]))
        axarr[i].plot(np.linspace(0,15,213),abs(tgp2[i]))
    axarr[0].set_title('Time lag spectrum of giant pulse at {}, {}'.format(time_values[index[0]],time_values[index[1]]))
    axarr[7].set_xlabel(r'Time lag(\mu s)')


# Function to compute the correlation coefficient

# In[5]:

# bgstd = np.load('./figures/correlation_coeff/backgroundstd.npy')

def find_cc((fs1,fs1bg),(fs2,fs2bg), binned = 0, bin_no=16):
    '''This find_cc function uses Robert\'s .npy pulses, with a different bg noise for each pulse''' 
    print fs1.shape
    fgp1=process_freq_spec(fs1,nchan,draw=0) #-1 if getting i from .txt file. i only needed for plotting
    fgp1bg=process_freq_spec(fs1bg,nchan,draw=0)
    tgp1=transform_to_timelag(fgp1,draw = 0)
    tgp1bg = transform_to_timelag(fgp1bg,draw = 0)
    print tgp1.shape
    fgp2=process_freq_spec(fs2,nchan,draw=0)
    tgp2=transform_to_timelag(fgp2,draw = 0)
    fgp2bg=process_freq_spec(fs2bg,nchan,draw=0)
    tgp2bg=transform_to_timelag(fgp2bg,draw = 0) 
    
    time_lag_values = np.linspace(0,15,213)
    if binned:
#         bins = np.logspace(np.log10(time_lag_values[1]),np.log10(15),bin_no)
        bins = np.linspace(0,15,bin_no)
        numerator_8 = np.zeros(shape = (0,bin_no-1))
    else:
        numerator_8 = np.zeros(shape = (0,213))
        
    for k in range(8):
        if binned:
            numerator = np.real(tgp1[k]*np.conj(tgp2[k]))
            numerator_8 = np.append(numerator_8,np.array([bin_data(time_lag_values,numerator,bins)]),0)
        else:
            numerator_8 = np.append(numerator_8,np.array([tgp1[k]*np.conj(tgp2[k])]),0)
    
    bgstd1=find_std(tgp1bg)
    bgstd2=find_std(tgp2bg)
#     print bgstd1[:10]
#     print bgstd2[:10]
    numerator = numerator_8.mean(0)
     
#     tgp1b = np.zeros(shape = (0,bin_no-1))
#     tgp2b = np.zeros(shape = (0,bin_no-1))
#     for k in range(8):
#         tgp1b = np.append(tgp1b,np.array([bin_data(time_lag_values,tgp1[k],bins)]),0)
#         tgp2b = np.append(tgp2b,np.array([bin_data(time_lag_values,tgp2[k],bins)]),0)
    
#     denominator = np.std(tgp1b)*np.std(tgp2b)
    
#     The old method of getting standard deviation
    stdgp1 = find_std(tgp1)
    print 'tgp1 has type', tgp1.dtype
    print 'stdgp1 has type',stdgp1.dtype
    stdgp2 = find_std(tgp2)
    if binned:
        stdgp1b = bin_data(time_lag_values,stdgp1,bins)
        stdgp2b = bin_data(time_lag_values,stdgp2,bins)
        bgstd1b = bin_data(time_lag_values,bgstd1,bins)
        bgstd2b = bin_data(time_lag_values,bgstd2,bins)
        denominator = np.sqrt((stdgp1b**2-bgstd1b**2)*(stdgp2b**2-bgstd2b**2))
#         denominator = stdgp1b*stdgp2b
#         denominator = bin_data(time_lag_values,np.std((tgp1*np.conj(tgp2)),axis=0),bins)
    else:
        denominator = np.sqrt((stdgp1**2-bgstd1**2)*(stdgp2**2-bgstd2**2))
#         denominator = stdgp1*stdgp2
#         denominator = np.std((tgp1*np.conj(tgp2)),axis=0)


    cc = numerator/denominator
    figure()
    if binned:
        plot(bins[:-1],denominator,label = 'denominator')
        plot(bins[:-1], numerator, label = 'numerator')
    else:
        plot(time_lag_values, numerator, label = 'numerator')
        plot(time_lag_values,denominator,label = 'denominator')
#     print 'denominator is ',denominator
#     title('Numerator and denominator of cc between {} {}'.format(tele1,))
    legend()
    return cc



# Linear binning

# In[159]:

tgp1


# In[12]:

i = 235
j = 85
# fs1ef = load_gp(text_lines[i-1],'ef',512)
# fs2ef = load_gp(text_lines[j-1],'ef',512)
# fs1jb = load_gp(text_lines[i-1],'jb',512)
# fs2jb = load_gp(text_lines[j-1],'jb',512)
time_string1 = '2015-10-19T00:17:47.415' # S/N = 89!, main pulse 
time_string2 = '2015-10-19T02:35:26.143' # S/N = 43, main pulse 
time_string3 = '2015-10-19T02:06:52.280' #S/N = 7.9, inter pulse
time_string4 = '2015-10-19T02:06:58.031' #S/N = 9.6, main pulse

time_string5 = '2015-10-18T23:47:20.315' #S/N = 11, interpulse
time_string6 = '2015-10-18T23:47:20.807' #S/N = 11, main pulse

time_string7 = '2015-10-19T00:37:14.418' #S/N = 13, main pulse
time_string8 = '2015-10-19T00:37:15.126' #S/N = 7.2, main pulse

S_N2 = 43
S_N1 = 89
#Using robert's data
ccefjb = find_cc(load_gp(time_string1,'ef',512),load_gp(time_string1,'jb',512),binned=1)
# ccjbwb = find_cc(load_gp(time_string1,'jb',512),load_gp(time_string1,'wb',512),binned=1)
# ccefwb = find_cc(load_gp(time_string1,'ef',512),load_gp(time_string1,'wb',512),binned=1)

ccnoisetest = find_cc((gp1,noise1),(gp2,noise2),binned = 1)
#Getting data from my frequency array 
# ccef1 = find_cc1(freq_values[i-1],freq_values[j-1],binned=1)


# In[74]:

'''Linear plotting'''
N=ccefjb.shape[0]

figure(figsize= (10,5))
ax1 = subplot(3,1,1)
plot(np.linspace(0,15,N),ccefjb,label='ef-ef')
# ylim(-.2,.2)
axhline(y=0.,color = 'black')
# title('correlation coefficient as a function of timelag of two main pulses\n at {}, {}'.format(time_values[i-1],time_values[j-1])) 
title('correlation coefficient of the same pulse with S/N per band of {} at {} between 2 telescopes. \n real part of numerator used.'.format(S_N1,time_string1))
legend(loc=(0.95,0.5))

# ax2=subplot(3,1,2)
# plot(np.linspace(0,15,N),ccjbwb,'c',label = 'jb-jb')
# # ylim(-.2,.2)
# axhline(y=0.,color = 'black')
# # axarr[1].set_title('jb:correlation coefficient as a function of timelag of two main pulses\n at {}, {}'.format(time_values[i-1],time_values[j-1])) 
# legend(loc=(0.95,0.5))

# ax2=subplot(3,1,3)
# plot(np.linspace(0,15,N),ccefwb,'g',label='wb-wb')
# # ylim(-.2,.2)
# axhline(y=0.,color = 'black')
# xlabel(r'time lag ($\mu$s)')
# # axarr[2].set_title('wb:correlation coefficient as a function of timelag of two main pulses\n at {}, {}'.format(time_values[i-1],time_values[j-1])) 
# legend(loc=(0.95,0.5))


# '''Log plotting'''
# figure(figsize=(10,3))
# semilogx(ccef)


# Log binning

# In[52]:

bin_no = 16
time_lag_values = np.linspace(0,15,213)
bins = np.logspace(np.log10(time_lag_values[1]),np.log10(15),bin_no)
cc_means = np.histogram(time_lag_values,bins,weights = cc,density = False)[0]/np.histogram(time_lag_values,bins,density = False)[0]

figure(figsize=(10,3))
semilogx(bins[:-1],cc_means)


# In[27]:

get_ipython().magic(u'pylab inline')
#load sigma noise
bgstd = np.load('./figures/correlation_coeff/backgroundstd.npy')

i = 34
j = 35
#def find_cc(i,j):
fgp1=process_freq_spec(freq_values[i-1],i,draw=0) #-1 if getting i from .txt file. i only needed for plotting
tgp1=transform_to_timelag(fgp1,i-1,draw = 0)
fgp2=process_freq_spec(freq_values[j-1],j,draw=0)
tgp2=transform_to_timelag(fgp2,j-1,draw = 0)

numerator_8 = np.zeros(shape = (0,213))
for k in range(8):
    numerator_8 = np.append(numerator_8,np.array([tgp1[k]*np.conj(tgp2[k])]),0)

numerator = np.real(numerator_8).mean(0)

stdgp1 = find_std(tgp1)
stdgp2 = find_std(tgp2)
denominator = np.sqrt((stdgp1**2-bgstd**2)*(stdgp2**2-bgstd**2))
    
cc = numerator/denominator
    #return cc

# cc = find_cc(34,35)


# In[ ]:

def find_cc1(fs1,fs2, binned = 0, bin_no=16):
    '''This find_cc function uses my original data freq_values, with the same background std'''
    bgstd = np.load('./figures/correlation_coeff/backgroundstd.npy')
    fgp1=process_freq_spec(fs1,draw=0) #-1 if getting i from .txt file. i only needed for plotting
    tgp1=transform_to_timelag(fgp1,draw = 0)
    fgp2=process_freq_spec(fs2,draw=0)
    tgp2=transform_to_timelag(fgp2,draw = 0)
    
    if binned:
        numerator_8 = np.zeros(shape = (0,bin_no-1))
    else:
        numerator_8 = np.zeros(shape = (0,213))
        
    for k in range(8):
        if binned:
            numerator = tgp1[k]*np.conj(tgp2[k])
            numerator_8 = np.append(numerator_8,np.array([bin_data(np.linspace(0,15,213),numerator,np.linspace(0,15,bin_no))]) ,0)
        else:
            numerator_8 = np.append(numerator_8,np.array([tgp1[k]*np.conj(tgp2[k])]),0)

    numerator = np.real(numerator_8).mean(0)
    
    stdgp1 = find_std(tgp1)
    stdgp2 = find_std(tgp2)
    if binned:
        stdgp1b = bin_data(np.linspace(0,15,213),stdgp1,np.linspace(0,15,bin_no))
        stdgp2b = bin_data(np.linspace(0,15,213),stdgp2,np.linspace(0,15,bin_no))
        bgstdb = bin_data(np.linspace(0,15,213), bgstd ,np.linspace(0,15,bin_no))
#         denominator = np.sqrt((stdgp1b**2-bgstdb**2)*(stdgp2b**2-bgstdb**2))
        denominator = np.sqrt((stdgp1b**2)*(stdgp2b**2))
    else:
#         denominator = np.sqrt((stdgp1**2-bgstd**2)*(stdgp2**2-bgstd**2))
        denominator = np.sqrt((stdgp1**2)*(stdgp2**2))

    cc = numerator/denominator
    return cc


# In[28]:

f,axarr=subplots(6,1,figsize=(10,21))
for k in range(8):
    axarr[0].plot(np.linspace(0,15,213),numerator_8[k])
axarr[0].set_title('numerator for each band plotted on same graph')

# figure(figsize=(10,3))
axarr[1].plot(np.linspace(0,15,213),numerator)
axarr[1].set_title('numerator meaned (average of above)')

# figure(figsize=(10,3))
axarr[2].plot(np.linspace(0,15,213),denominator)
axarr[2].set_title('denominator (standard deviations)')

# figure(figsize=(10,3))
axarr[3].plot(np.linspace(0,15,213),cc)
axarr[3].set_title('correlation coefficient as a function of timelag of two main pulses\n at {}, {}'.format(time_values[i-1],time_values[j-1])) 

axarr[4].plot(np.linspace(0,15,213),bgstd)
# axarr[4].set_xlabel(r'time lag ($\mu$ s)')
axarr[4].set_title('standard deviation of background frequency') 

axarr[5].plot(np.linspace(0,15,213),stdgp1)
axarr[5].plot(np.linspace(0,15,213),stdgp2)
axarr[5].set_xlabel(r'time lag ($\mu$ s)')
axarr[5].set_title('standard deviation of the two pulses') 

figure(figsize=(10,3))
plot(np.linspace(0,15,213),cc)
ylim(-1.,1.)
title('correlation coefficient as a function of timelag of two main pulses\n at {}, {}'.format(time_values[i-1],time_values[j-1])) 


# In[32]:

a= np.array([1,2,3])
a**2
numerator_8.shape


# Plot the frequency spectrum and time lag spectrum of two giant pulses

# In[54]:

get_ipython().magic(u'pylab inline')
# i=34
# j=35

fgp1=process_freq_spec(freq_values[i-1],i-1,draw=0)
tgp1=transform_to_timelag(fgp1,i)

fgp2=process_freq_spec(freq_values[j-1],j-1,draw=0)
tgp2=transform_to_timelag(fgp2,j)

plot_2_freq(fgp1,fgp2,[i-1,j-1])
plot_2_timelag(tgp1,tgp2,[i-1,j-1])


# In[1]:

tgp66


# In[59]:

abs(tgp10).shape


