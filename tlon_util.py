import pickle
def read_tlon(idir,q,r,suffix):
    qstr = str(q)
    rstr = str(r)
    infile = idir+'/q'+qstr+'_r'+rstr+'_'+suffix
    with open(infile, 'rb') as file:
    # Load the data from the pickle file
        data_tuple = pickle.load(file)
    return data_tuple
def read_time_lon(infile):
    with open(infile, 'rb') as file:
    # Load the data from the pickle file
        data_tuple = pickle.load(file)
    return data_tuple

def read_tlon(infile):
    with open(infile, 'rb') as file:
    # Load the data from the pickle file
        data_tuple = pickle.load(file)
    return data_tuple
def compute_rms(data):
    rms = numpy.sqrt(numpy.mean(data**2))
    return rms
def norm(arr):
    return arr/numpy.max(arr)

def filter1d(f,g,time):
    import numpy.fft as fft
    # returns ifft(fft(f)*g)
    # f(time)  : 1-d function to be filtered
    # time     : 1-d array of time values
    # g(omega) : filter function, defined on omega grid corresponding to time array
    #          : note that omega=0 is assumed to sit at array element len(time)//2
    f_fft, omega = my_fft_1d(f,time)
    filtered_fft = f_fft*g
    dshift = fft.ifftshift(filtered_fft)
    filtered_f=fft.ifft(dshift)
    return filtered_f
    
def view_tlon(tlon_in,time,lon,ax,remove_mean=True,scale_factor=1.5,
              color_map='RdYlBu_r',time_units='',q_units='',fig=None,
              reference_slope=None, tmax=None):

    tlon = tlon_in[:,:]
    nt = len(time)


    if (remove_mean):
        print('Removing Mean')
        for k in range(nt):
            tlon[k,:] = tlon[k,:]-numpy.mean(tlon[k,:])    

    rms = compute_rms(tlon)
    mini = -scale_factor*rms
    maxi = scale_factor*rms        
    
    exts = [0, 360, time[0],time[nt-1]]
    
    im = ax.imshow(tlon,origin='lower',aspect='auto',vmin=mini,vmax=maxi,cmap=color_map,extent=exts)
    ax.set_xlabel('Longitude')
    tlabel = 'Time'
    if (time_units != ''):
        tlabel=tlabel+' ('+time_units+')'
    ax.set_ylabel(tlabel)    
    
    if (tmax == None):
        tmax = time[nt-1]
    
    if (reference_slope != None):
        tmid = 0.5*(tmax-time[0])+time[0]
        degmid = 180
        nx = len(tlon[0,:])
        x = numpy.arange(nx)*360/nx
        y = (lon-degmid)*reference_slope+tmid
        ax.plot(x,y)
    
    if (fig != None):
        fig.colorbar(im,ax=ax, label=q_units,pad=0.08,location='bottom')

    ax.set_ylim([time[0],tmax])
