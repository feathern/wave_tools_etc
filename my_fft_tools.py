def my_fft_1d(data,time,apodize=True,apod_xp=0.1,apod_xs=30):
    import numpy.fft as fft
    from numpy import pi
    import numpy as np
    # Returns a tuple (fft(data),omega)
    # The fft and the omega array are returns in order
    # of ascending omega (i.e., human order, not fft order)
    nt = len(time)
    dt = time[1]-time[0]
    tlen=dt*nt
    
    
    domega=2*pi/tlen
    omega = np.zeros(nt,dtype='float64')

    for i in range(nt):
        omega[i]=domega*(i-nt//2)

    #########################
    # Create apodization function
    nx = len(data)
    xnorm = np.linspace(0,1,nx,dtype='float64')
    xp = apod_xp
    xs=apod_xs
    xtan=(xnorm-xp)*xs
    xtan2=(xnorm-(1-xp))*xs
    apod1 = (np.tanh(xtan)+1)*0.5
    apod2 = 1-(np.tanh(xtan2)+1)*0.5
    apod = (apod1+apod2)-1
    
    dfft=fft.fft(data*apod)
    dshift = fft.fftshift(dfft)   # This shifts 0 frequency to element nt//2
    
    return (dshift,omega)

def my_acorr_1d(data,time):
    import numpy.fft as fft
    import numpy as np

    nt = len(time)
    dt = time[1]-time[0]

    dshift, omega = my_fft_1d(data,time)
    
    power = dshift.real**2+dshift.imag**2
    pshift = fft.ifftshift(power)
    ccor = fft.fft(pshift)
    ccor = fft.fftshift(ccor).real

    tcor = np.zeros(nt,dtype='float64')
    for i in range(nt):
        tcor[i]=dt*(i-nt//2)

    return (ccor,tcor)

def my_fft_2d(data,time,x,rev_omega=True, shift_data=True, compute_freq = True):
    import numpy.fft as fft
    from numpy import pi
    import numpy as np
    # Returns a tuple (fft(data),omega)
    # The fft and the omega array are returns in order
    # of ascending omega (i.e., human order, not fft order)
    nt = len(time)
    dt = time[1]-time[0]
    tlen=dt*nt
    

    
    domega=2*pi/tlen
    omega = np.zeros(nt,dtype='float64')

    for i in range(nt):
        omega[i]=domega*(i-nt//2)


    nx = len(x)
    dx = x[1]-x[0]
    xlen=dx*nx

    
    dm=2*pi/xlen
    m = np.zeros(nx,dtype='float64')

    for i in range(nx):
        m[i]=dm*(i-nx//2)

    dfft=fft.fft2(data)
    if (shift_data):
        dshift = fft.fftshift(dfft)   # This shifts 0 frequency to element nt//2
    
    if (rev_omega):
        dshift = dshift[::-1,:]  # LEFT OFF HERE -- think on omega
        omega = -omega[::-1]
    
    return (dshift,omega,m)

def my_ccorr_1d(data1, data2, time):
    import numpy.fft as fft
    import numpy as np
    # Returns cross correlation of data1 and data 2 defined as
    # c[d] = sum_i data1_i(i+d) data2_i
    # c[d] is computed using Fourier transforms
    
    nt = len(time)
    dt = time[1]-time[0]

    dshift1, omega = my_fft_1d(data1,time)
    dshift2, omega = my_fft_1d(data2,time)
    
    power = dshift1*0
    power.real = dshift1.real*dshift2.real +dshift1.imag*dshift2.imag
    power.imag = -dshift1.real*dshift2.imag+dshift2.real*dshift1.imag
    
    pshift = fft.ifftshift(power)
    ccor = fft.fft(pshift)
    mxreal = np.max(np.abs(ccor.real))
    mximag = np.max(np.abs(ccor.imag))
    
    print('Check: ', mxreal, mximag)
    ccor = fft.fftshift(ccor).real
    ccor = ccor/nt # Adjust for FFT normalization convention
    
    tcor = np.zeros(nt,dtype='float64')
    for i in range(nt):
        tcor[i]=dt*(i-nt//2)

    return (ccor,tcor)

def my_ccorr_2d(data1, data2, dx1,dx2):
    import numpy.fft as fft
    import numpy as np
    # Returns 2D cross correlation of data1 and data 2 defined as
    # c[n,m] = sum_ij { data1_ij(i+n,j+m) data2_ij }
    # c[n,m] is computed using Fourier transforms
    
    nx1=data1.shape[0]
    nx2=data1.shape[1]

    dshift1=fft.fft2(data1)
    dshift2=fft.fft2(data2)
    
    power = dshift1*0
    power.real = dshift1.real*dshift2.real +dshift1.imag*dshift2.imag
    power.imag = -dshift1.real*dshift2.imag+dshift2.real*dshift1.imag
    
    ccor = fft.fft2(power)    

    ccor = fft.fftshift(ccor).real
    ccor = ccor/(nx1*nx2) # Adjust for FFT normalization convention
    
    xcor1 = np.zeros(nx1,dtype='float64')
    for i in range(nx1):
        xcor1[i]=dx1*(i-nx1//2)

    xcor2 = np.zeros(nx2,dtype='float64')
    for i in range(nx2):
        xcor2[i]=dx2*(i-nx2//2)    
        
    return (ccor,xcor1,xcor2)