import numpy as np
def sep_sort(inlist,mval,offset=0):
    vpos=[]  # postive values
    vneg=[]  # negative values
    ipos = [] # index corresponding to original index of positive values
    ineg = [] # and that for the negative values
    
    for i, v in enumerate(inlist):
        if (v <= 0):
            vneg.append(v)
            ineg.append(i+offset)
        else:
            vpos.append(v)
            ipos.append(i+offset)
    apos = np.array(vpos)
    iipos = np.array(ipos)
    sort_indices = np.argsort(-apos)
    ipos = iipos[sort_indices]
    vpos = apos[sort_indices]
    
    aneg = np.array(vneg)
    iineg = np.array(ineg)
    sort_indices = np.argsort(aneg)
    ineg = iineg[sort_indices]
    vneg = aneg[sort_indices]
    
    pos_str = []
    neg_str = []
    for i in ipos:
        pos_str.append('v '+str(i)+'\n B '+str(i+mval))
    for i in ineg:
        neg_str.append('v '+str(i)+'\n B '+str(i+mval))
    
    return vpos, vneg, ipos, ineg, pos_str, neg_str


def bplot(arr,ax,mval,nplot=5,margin=0.2):

    offset = -(len(arr)-1)//2
    vpos, vneg, ipos, ineg, pos_str, neg_str = sep_sort(arr,mval,offset=offset)

    npos = len(ipos)
    nneg = len(ineg)

    empty = []

    for i in range(npos+nneg):
        empty.append(' '*(i+1))

    bar_width = 0.9

    #nplot = 10

    p = ax.bar(empty[0:nplot], vpos[0:nplot],edgecolor='black', width=bar_width, color='tab:red') 
    ax.bar_label(p, labels=pos_str[0:nplot],label_type='edge',padding=4)    

    p = ax.bar(empty[0:nplot], vneg[0:nplot],edgecolor='black', width=bar_width, color='tab:blue') 
    test = ax.bar_label(p, labels=neg_str[0:nplot],label_type='edge',padding=4)    
    
    span = vpos[0]-vneg[0]
    
    
    
    ymax = vpos[0]+span*margin
    ymin = vneg[0]-span*margin

    
    ax.set_ylim([ymin,ymax])
    

def quad_plot(emfc,enrmc,mval,icomp,ax,nplot=5):
    vtBp = r'$v_\theta B_\phi$'
    vpBt = r'$v_\phi B_\theta$'
    
    vpBr = r'$v_\phi B_r$'
    vrBp = r'$v_r B_\phi$'
    
    vrBt = r'$v_r B_\theta$'
    vtBr = r'$v_\theta B_r$'
    
    #Emf_phi  = vr B_theta - vtheta B_r
    
    er_names = [vtBp, vpBt]
    et_names = [vpBr, vrBp]
    ep_names = [vrBt, vtBr]
    
    r_names = [et_names, ep_names]
    t_names = [er_names, ep_names]
    p_names = [er_names, et_names]
    
    all_names = [r_names, t_names, p_names]
    
    names = all_names[icomp]
    
    all_inds = [ [1,2], [0,2], [0,1] ]
    
    inds = all_inds[icomp]
    
    for i in range(2):
        for j in range(2):
            nrm=enrmc[inds[i]][mval]/np.max(enrmc)
            bplot(emfc[inds[i]][j,mval,:]*nrm, ax[i][j],mval,nplot=nplot)
            ax[i][j].set_title(names[i][j])
    #bplot(ax[0][1], emfc[inds[0][1,mval,:])
     
    #bplot(ax[1][0], emfc[inds[1][0,mval,:])
                         
    #bplot(ax[1][1], emfc[inds[1][1,mval,:])
                         
def build_emf_correlations0(data_tuple,iv1,ib1,iv2,ib2,remove_dr=False,mmax=32,mfil_max=33,refilter=False, mstart=0):
    # Total emf = v1*b1-v2*b2
    # iv1,ib1,iv2,ib2 indicate the indices of v and b to use 
    # (0,1,2 = vr,vtheta,phi; 3,4,5 = br,btheta,bphi)
    
    nm_fil = mfil_max-mstart+1
    print('hmm: ', mfil_max, mstart, nm_fil)
    # Make a copy since we are going to do some filtering
    bv_fft = data_tuple[0].copy()

    #Note that bv_fft has not been shifted, but omega and mvals have
    omega = data_tuple[1].copy()
    mvals = data_tuple[2].copy()
    bv = data_tuple[3]
    nm = len(mvals)
    nt = len(omega)
    print('nm: ', nm)


    vis=np.arange(-mmax,mmax+1)
    bv_fft = np.fft.fftshift(bv_fft,axes=(2))

    # Remove the differential rotation if desired
    if (remove_dr):
        vind = 0
        if (iv1 == 2):
            vind = iv1
        if (iv2 == 2):
            vind = iv2
        if (vind == 2):
            for i in range(nt):
                bv[vind,i,:] = bv[vind,i,:]-numpy.mean(bv[2,i,:])


    emf = bv[iv1,:,:]*bv[ib1,:,:] - bv[iv2,:,:]*bv[ib2,:,:] # The full emf for this vB pair -- unfiltered
    emf_fft = np.fft.fft2(emf) # And its FFT
    emf_fft = np.fft.fftshift(emf_fft,axes=(1)) # shifted so m=0 is centered
    emf_fil_fft = 0*emf_fft

    if (refilter==True):
        p1_fil_fft = 0*emf_fft
        p2_fil_fft = 0*emf_fft

    print('nm_fil: ', nm_fil)
    pcorr = numpy.zeros((3,nm_fil,2*mmax+1),dtype='float64')
    enrm_save = numpy.zeros(nm_fil)

    for j in range(nm_fil):
        mfil = j+mstart
        print('Filtering for m = ', mfil)
        # First, filter the emf
        emf_fil_fft[:,:] = 0+0j 
        emf_fil_fft[:,nm//2+mfil] = emf_fft[:,nm//2+mfil]
        emf_fil_fft[:,nm//2-mfil] = emf_fft[:,nm//2-mfil]
        emf_fil_fft = np.fft.fftshift(emf_fil_fft,axes=(1))
        emf_fil = np.fft.ifft2(emf_fil_fft).real  # This emf is filtered for the m under consideration
        enrm = numpy.sum(emf_fil*emf_fil)
        enrm_save[j] = enrm
        
        bis = vis+mfil
        for i, vi in enumerate(vis):

            bi = bis[i]

            v1_fft = 0*bv_fft[0,:,:]
            b1_fft = 0*v1_fft
            
            v2_fft = 0*v1_fft
            b2_fft = 0*v2_fft

            v1_fft[:,nm//2+vi] =  bv_fft[iv1,:,nm//2+vi]
            v2_fft[:,nm//2+vi] =  bv_fft[iv2,:,nm//2+vi]
            
            b1_fft[:,nm//2+bi] =  bv_fft[ib1,:,nm//2+bi]
            b2_fft[:,nm//2+bi] =  bv_fft[ib2,:,nm//2+bi]

            v1_fft[:,nm//2-vi] =  bv_fft[iv1,:,nm//2-vi]
            v2_fft[:,nm//2-vi] =  bv_fft[iv2,:,nm//2-vi]
            
            b1_fft[:,nm//2-bi] =  bv_fft[ib1,:,nm//2-bi]
            b2_fft[:,nm//2-bi] =  bv_fft[ib2,:,nm//2-bi]
            
            

            v1_fft = np.fft.fftshift(v1_fft,axes=(1))
            v2_fft = np.fft.fftshift(v2_fft,axes=(1))            
            
            b1_fft = np.fft.fftshift(b1_fft,axes=(1))
            b2_fft = np.fft.fftshift(b2_fft,axes=(1))       
            
            # Left off here (Aug 22, 11:28)
            
            if (remove_dr and (vi==0)):
                if (iv1 == 2):
                    v1_fft = 0*v1_fft
                if (iv2 == 2):
                    v2_fft = 0*v2_fft



            v1 = np.fft.ifft2(v1_fft).real   
            v2 = np.fft.ifft2(v2_fft).real  
            b1 = np.fft.ifft2(b1_fft).real   
            b2 = np.fft.ifft2(b2_fft).real
                
            label = 'b (m = '+str(bi)+') ;  v (m = '+str(vi)+')'
            p1 = v1*b1
            p2 = -v2*b2
            
            ##################################
            # Probably need to filter p1 and p2 for m = mfil due to how things were done above...
            if (refilter==True):
                p1_fft = np.fft.fft2(p1) 
                p1_fft = np.fft.fftshift(p1_fft,axes=(1)) # shifted so m=0 is centered
                
                p2_fft = np.fft.fft2(p2) 
                p2_fft = np.fft.fftshift(p2_fft,axes=(1)) # shifted so m=0 is centered
                
                p1_fil_fft[:,:] = 0+0j
                p2_fil_fft[:,:] = 0+0j
                
                p1_fil_fft[:,nm//2+mfil] = p1_fft[:,nm//2+mfil]
                p1_fil_fft[:,nm//2-mfil] = p1_fft[:,nm//2-mfil]
                
                p2_fil_fft[:,nm//2+mfil] = p2_fft[:,nm//2+mfil]
                p2_fil_fft[:,nm//2-mfil] = p2_fft[:,nm//2-mfil]                
                
                p1_fil_fft = np.fft.fftshift(p1_fil_fft,axes=(1))
                p1 = np.fft.ifft2(p1_fil_fft).real  
                
                p2_fil_fft = np.fft.fftshift(p2_fil_fft,axes=(1))
                p2 = np.fft.ifft2(p2_fil_fft).real                
            
            ptot = p1+p2
            if (i%16==0):
                print(label)


            pcorr[0,j,i] = numpy.sum(p1*emf_fil)/enrm
            pcorr[1,j,i] = numpy.sum(p2*emf_fil)/enrm
            p1+=p2
            pcorr[2,j,i] = numpy.sum(p1*emf_fil)/enrm    
        
    return pcorr, enrm_save

def build_emf_correlations(v1,b1,v2,b2,mfil_min=0, mfil_max=33,mmax=32,refilter=False):
    # computes correlation matrix for  v1*b1-v2*b2
    # v1,v2,b1,b2 are dimensioned [time,space]
    # 
    # mfil_min:  the minimum m-value to filter for
    # mfil_max:  the maximum m-value to filter for
    # mmax :  the maximum absolute value of m to consider 
    #         the range of m-values considered is [-mmax : mmax ]
    # Returns correlation matrix dimensioned:  [3,0:mfil_max-mfil_min+1, 0: 1+2*mmax]
    # index 0:  projection of v1*b1 at given m1,m2 onto filtered v1b2-v2*b2
    # index 1:  same, but for -v2b2
    # index 2:  same, but for full v1b1-v2b2 at given m
    
    nm_fil = mfil_max-mfil_min+1
    print('hmm: ', mfil_max, mstart, nm_fil)

    nt = v1.shape[0]
    nx = v1.shape[1]
    nm = nx

    vis=np.arange(-mmax,mmax+1)

    # Compute the Full EMF, its FFT, and the FFTs of v1,B1, v2 and B2
    emf = v1[:,:]*b1[:,:] - v2[:,:]*b2[:,:] # The full emf for this vB pair -- unfiltered
    emf_fft = np.fft.fft2(emf) # And its FFT
    emf_fil_fft = 0*emf_fft

    v1_fft = np.fft.fft2(v1)
    v2_fft = np.fft.fft2(v2)
    b1_fft = np.fft.fft2(b1)
    b2_fft = np.fft.fft2(b2)
    
    v1_fil_fft = 0*v1_fft
    v2_fil_fft = 0*v2_fft    
    b1_fil_fft = 0*b1_fft
    b2_fil_fft = 0*b2_fft        
    
    if (refilter==True):
        p1_fil_fft = 0*emf_fft
        p2_fil_fft = 0*emf_fft

    pcorr = numpy.zeros((3,nm_fil,2*mmax+1),dtype='float64')
    enrm_save = numpy.zeros(nm_fil)

    for j in range(nm_fil):
        mfil = j+mfil_min
        print('Filtering for m = ', mfil)
        # First, filter the emf
        emf_fil_fft[:,:] = 0+0j 
        emf_fil_fft[:,mfil] = emf_fft[:,nm-mfil]
        emf_fil_fft[:,mfil] = emf_fft[:,nm-mfil]
        emf_fil = np.fft.ifft2(emf_fil_fft).real  # This emf is filtered for the m under consideration

        enrm = numpy.sum(emf_fil*emf_fil)
        enrm_save[j] = enrm
        
        bis = vis+mfil
        for i, vi in enumerate(vis):

            bi = bis[i]

            v1_fil_fft[:,:] = 0+0j
            v2_fil_fft[:,:] = 0+0j
            b1_fil_fft[:,:] = 0+0j
            b2_fil_fft[:,:] = 0+0j
            

            # Copy spectra for v and B at wavenumbers vi and bi respectively
            v1_fil_fft[:,vi] =  v1_fft[:,vi]
            v2_fil_fft[:,vi] =  v2_fft[:,vi]
            
            v1_fil_fft[:,nm-vi] =  v1_fft[:,nm-vi]
            v2_fil_fft[:,nm-vi] =  v2_fft[:,nm-vi]
            
            b1_fil_fft[:,bi] =  b1_fft[:,bi]
            b2_fil_fft[:,bi] =  b2_fft[:,bi]
            
            b1_fil_fft[:,nm-bi] =  b1_fft[:,nm-bi]
            b2_fil_fft[:,nm-bi] =  b2_fft[:,nm-bi]            
                        
            #Compute contribution of combo vi, bi to the emf
                
            v1_fil = np.fft.ifft2(v1_fil_fft).real   
            v2_fil = np.fft.ifft2(v2_fil_fft).real  
            b1_fil = np.fft.ifft2(b1_fil_fft).real   
            b2_fil = np.fft.ifft2(b2_fil_fft).real
                
            label = 'b (m = '+str(bi)+') ;  v (m = '+str(vi)+')'
            
           
            p1 = v1_fil*b1_fil
            p2 = -v2_fil*b2_fil
            
            ##################################
            # Probably need to filter p1 and p2 for m = mfil due to how things were done above...
            if (refilter==True):
                p1_fft = np.fft.fft2(p1) 
                p1_fft = np.fft.fftshift(p1_fft,axes=(1)) # shifted so m=0 is centered
                
                p2_fft = np.fft.fft2(p2) 
                p2_fft = np.fft.fftshift(p2_fft,axes=(1)) # shifted so m=0 is centered
                
                p1_fil_fft[:,:] = 0+0j
                p2_fil_fft[:,:] = 0+0j
                
                p1_fil_fft[:,nm//2+mfil] = p1_fft[:,nm//2+mfil]
                p1_fil_fft[:,nm//2-mfil] = p1_fft[:,nm//2-mfil]
                
                p2_fil_fft[:,nm//2+mfil] = p2_fft[:,nm//2+mfil]
                p2_fil_fft[:,nm//2-mfil] = p2_fft[:,nm//2-mfil]                
                
                p1_fil_fft = np.fft.fftshift(p1_fil_fft,axes=(1))
                p1 = np.fft.ifft2(p1_fil_fft).real  
                
                p2_fil_fft = np.fft.fftshift(p2_fil_fft,axes=(1))
                p2 = np.fft.ifft2(p2_fil_fft).real                
            
            ptot = p1+p2
            if (i%16==0):
                print(label)


            pcorr[0,j,i] = numpy.sum(p1*emf_fil)/enrm
            pcorr[1,j,i] = numpy.sum(p2*emf_fil)/enrm
            p1+=p2
            pcorr[2,j,i] = numpy.sum(p1*emf_fil)/enrm    
        
    return pcorr, enrm_save


def call_build_emf_example():
    # Just for reference
    import numpy as np
    import numpy
    import matplotlib.pyplot as plt
    import pickle
    rind = 1
    lind = 2

    fft_file = 'FFT/bv_r'+str(rind)+'_l'+str(lind)
    with open(fft_file, 'rb') as file:
    # Load the data from the pickle file
        data_tuple = pickle.load(file)

    mmax=70
    mfmax=70
    mfmax = -4
    mstart = -4
    #mmax=4
    #mfmax=4
    # Emf_r  = vtheta B_phi - vphi B_theta
    iv1 = 1 # vtheta
    ib1 = 5 # Bphi
    iv2 = 2 # vphi
    ib2 = 4 # Btheta
    pcorr_r, enrm_r = build_emf(data_tuple,iv1,ib1,iv2,ib2,remove_dr=False,
                                mstart=mstart,mmax=mmax,mfil_max=mfmax, refilter = True)

    # Emf_theta  = vphi B_r - vr B_phi
    iv1 = 2 # vphi
    ib1 = 3 # Br
    iv2 = 0 # vr
    ib2 = 5 # Bphi
    pcorr_t, enrm_t = build_emf(data_tuple,iv1,ib1,iv2,ib2,remove_dr=False,
                                mstart=mstart,mmax=mmax,mfil_max=mfmax, refilter = True)

    # Emf_phi  = vr B_theta - vtheta B_r
    iv1 = 0 # vr
    ib1 = 4 # Btheta
    iv2 = 1 # vtheta
    ib2 = 3 # Br
    pcorr_p, enrm_p = build_emf(data_tuple,iv1,ib1,iv2,ib2,remove_dr=False,
                                mstart=mstart,mmax=mmax,mfil_max=mfmax, refilter = True)

    mvals=np.arange(-mmax,mmax+1)
    emf_file='emf_filter_matrices_new_fil_minus.dat'
    with open(emf_file,'wb') as myfile:
        pickle.dump((mvals,pcorr_r, enrm_r,pcorr_t, enrm_t,pcorr_p, enrm_p,fft_file),myfile)
    
