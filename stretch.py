# this file contains a number of functions for stretching waveforms that can be used to determine changes in seismic velocity for example
# "AA_stretch", or "adaptive alignment stretch", is described in Merrill et al. (2023)

def differentiate(fn,dt):
    import numpy as np
    import scipy as sp

    # Expect seismogram sections with time aligned along axis = 1
    m,n=fn.shape   
    n2=n//2+1
    sfn=sp.fft.irfft(sp.fft.rfft(fn)*np.outer(np.ones(m),(1j*np.array(np.arange(n2)*2*np.pi/(n*dt)))),n)
    return sfn

def base_stretch(trace,npts):
    import numpy as np
    # stretch a trace by a npts by inserting zeros in the fourier domain
    # equivalent to sinc interpolation
    TR  = np.fft.fft(trace)
    pos = TR[:int(len(TR)/2)]
    neg = TR[int(len(TR)/2):]
    
    C = np.zeros((npts,1),dtype = np.complex_)

    TRst = np.concatenate((pos, C[:,0], neg))
    
    sttrace = np.real(np.fft.ifft(TRst)) 
    return sttrace

# "Percent Stretch" uses the numpy interp function as removing zeros from the fourier transform and ifft-ing results in 
def perc_stretch4(trace,EC,tracelen):
    import numpy as np
    # stretch a trace by an expansion coefficient (EC; 1=no stretch)
    # tracelen is the length of the trace in seconds [s] 
    nzeros = round((len(trace) * (EC)) - len(trace))

    # need to define unstretched time vector
    tvec = np.linspace(0,tracelen,trace.shape[0])

    # define new time vector according to how many points need to be added/removed
    intvec = np.linspace(0,tracelen,trace.shape[0] + nzeros)
    sttrace = np.interp(x=intvec, xp=tvec, fp=trace)
        
    return sttrace

def AA_stretch(section,trace_len,int_mult=1):
    # waveform stretching function that maximizes the similarity of a section of waveforms (section) aligned at time zero
    # this function is based off the adaptive alignment iterative procedure described in Bostock et al. (2021) for aligning S-waves
    # three supporting functions (differentiate, perc_stretch4, and base_stretch) are contained in this py file

    ## -- INPUTS
    # section - a section of waveforms aligned at time zero as a numpy array
    # trace_len - length of section traces in seconds
    # int_mult - upsampling paramater to increase resolution (optional), recommend = 1 or 2. 

    ## -- OUTPUTS
    # stretch factors (s_factors)
    # updated section of waveforms (st_section)
    # and final objective function (eob_out)

    import numpy as np
    from scipy.linalg import svd

    ### --- Interpolate section of waveforms to improve resolution (optional) 
    base_len = section.shape[1]     # length of traces in samples
    st_section = []
    for itrace in range(section.shape[0]):
        st_trace = base_stretch(section[itrace,:],int_mult*base_len)
        st_section.append(st_trace)
    Traces_int = np.array(st_section)
    
    st_section = Traces_int                                # define st_section in case it doesn't get created in the loop
    tvec = np.linspace(0,trace_len,Traces_int.shape[1])    # define a time vector using interpolated section
    edt = tvec[1] - tvec[0]                                # sample interval [s]

    ### --- Initialize iteration parameters
    ns,_    = Traces_int.shape
    nit     = 500
    iter    = 0
    de      = 1.0
    tshift  = np.zeros(ns)
    tshift_running=np.ones(ns)
    tsr_old = tshift_running

    # "scomp" contains the section of waveforms being stretched. Remove mean and normalize.
    scomp   = np.zeros(Traces_int.shape) 
    for ix in np.arange(ns):
        scomp[ix,]=Traces_int[ix,]-np.mean(Traces_int[ix,])
        scomp[ix,]=scomp[ix,]/np.linalg.norm(scomp[ix,])

    # Enter iteration loop.
    while de > 0 and iter < nit:

        iter += 1
        U,s,Vh = svd(scomp)
        # compute starting objective function (phi = s1**2/n)
        eob_old=s[0]**2 / ns

        # first PC representation of section "scomp1"
        scomp1 = np.zeros(scomp.shape)
        scomp1 = scomp1+s[0]*np.outer(U[:,0],Vh[0,:])

        tvec   = np.linspace(0,trace_len,scomp.shape[1])
        edt    = tvec[1] - tvec[0]
        # Differentiate waveforms: 2 options - differentiate full waveform or its 
        # rank 2 approximation. The first can converge more slowly and sometimes 
        # achieves better objective but not always. Split difference by averaging the two.
        # Because for stretching we consider waveforms in the log-time domain, we need to apply
        # the chain rule for taking the derivative:
        # df/d(tau) = df/dt * dt/d(tau)
        # where dt/d(tau) is just the time vector
        scomp1d = (differentiate(scomp+scomp1,edt))/2 * tvec

        # Create perturbation from higher PC's (>2), resembling derivative.
        scomp2N = scomp-scomp1
        
        # Compute estimate of logarithmic time shifts.
        for i in range(ns):
            tshift[i]=np.dot(scomp1d[i,],scomp2N[i,])/np.dot(scomp1d[i,],scomp1d[i,])

        tshift         = tshift-np.mean(tshift)     # Apply zero sum constraint 
        tstretch       = np.exp(tshift)             # convert log-shifts into stretch factors
        tshift_running = tshift_running * tstretch  # keep track of cumulative stretch factors

        # pad_len is defined to pad to waveform section with zeros to the longest trace
        nzeros  = np.round((scomp.shape[1]*tstretch) - scomp.shape[1])
        pad_len = int(np.max(nzeros) + scomp.shape[1])

        # create a stretched section of waveforms from tstretch and define as "st_section"
        st_section = []
        for itrace in range(scomp.shape[0]):
            # stretch waveforms according to tshift
            st_trace = perc_stretch4(scomp[itrace,:],tstretch[itrace],trace_len)
            strP     = np.pad(st_trace,(0,pad_len - len(st_trace)),'constant')
            st_section.append(strP)
        
        # normalize to ensure objective functions are properly compared
        st_section = np.array(st_section)
        for ix in np.arange(ns):
            st_section[ix,]=st_section[ix,]-np.mean(st_section[ix,])
            st_section[ix,]=st_section[ix,]/np.linalg.norm(st_section[ix,])
        
        # compute updated objective function
        U,s,Vh = svd(st_section)
        eob_new = s[0]**2 / ns

        print('Iteration #', iter,eob_old,eob_new)

        # if the section has increased similarity, eob_new > eob_old and de will be positive
        de      = eob_new-eob_old
        eob_out = eob_new

        # revert to the last section if there is no improvement, otherwise update and continue
        if de <= 0: 
            tshift_running = tsr_old 
            st_section     = scomp
            eob_out        = eob_old
        else:
            tsr_old        = tshift_running
            print('Updating scomp')
            scomp          = st_section

    s_factors = tshift_running

    return s_factors,st_section,eob_out