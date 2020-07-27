import random
import time
import sys
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pylab as plt
import numpy as np
import cupy as cp
from math import ceil

class model_stim():

    def __init__(self, T, N, N_S, alpha=0.01, beta=0.01, alpha2=0.01, beta2=0.01, sparsity=0.01, mu=0, sigma=0.1):
        ''' Initialize class parameters and ground truth matrix
        '''
        self.T = T  # number of trials
        self.N = N   # number of neurons
        self.N_S = N_S  # number of neurons to stim each time
        
        self.alpha_rate = alpha    # actual test error rates
        self.beta_rate = beta

        self.alpha_rate2 = alpha2  # assumed test error rates
        self.beta_rate2 = beta2

        self.mu = mu
        self.sigma = sigma

        sparse = np.ones((N,N))*sparsity
        self.w0 = np.array(np.random.binomial(n=1, p=sparse), dtype='int')
        np.fill_diagonal(self.w0,0)
 
    def data_gen(self):
        ''' Generates data from the model
            Does one-at-a-time or N_S iid random selection
            Computes yhat, y, ahat, and c alongside s
        '''
        T = self.T
        N = self.N

        self.c = np.zeros((N,T), dtype=np.float32)
        self.s = np.zeros((N,T), dtype='int')
        self.y = np.zeros((N,T), dtype='int')
        self.ahat = np.zeros((N,T))
        self.yhat = np.zeros((N,T), dtype='int')

        ## Each neuron iid randomly selected for stimulation, total ~self.N_S
        self.s = np.random.binomial(1, (self.N_S/self.N), size=((self.N,T)))
        
        # ## OR force single stim at each step
        # stim = np.random.randint(self.N, size=(self.T))
        # self.s[stim, np.arange(0,self.T)] = 1

        self.yhat = np.max(np.multiply(self.s[np.newaxis,:,:],self.w0[...,np.newaxis]).astype(np.int), axis=1)
        zero_yhat = self.yhat == 0

        alpha_error = np.random.binomial(1,self.alpha_rate, size=(self.yhat[zero_yhat].shape))
        self.y[zero_yhat] = np.bitwise_or(self.yhat[zero_yhat], alpha_error).astype(np.int)
        beta_error = np.random.binomial(1,1-self.beta_rate, size=(self.yhat[~zero_yhat].shape))
        self.y[~zero_yhat] = np.multiply(self.yhat[~zero_yhat], beta_error).astype(np.int)

        self.ahat = self.yhat

        self.c = (np.log((1-self.alpha_rate2)*(1-self.beta_rate2)/(self.alpha_rate2*self.beta_rate2))*self.y - \
                        np.log((1-self.alpha_rate2)/self.beta_rate2)).astype('float32')

        # # For loop generation if N,T too large to fit in available memory
        # for t in range(self.T):
            
        #     # compute deterministic yhat
        #     self.yhat[:, t] = np.max(np.multiply(self.s[:,t], self.w0).astype(np.int), axis=1)

        #     zero_yhat = np.argwhere(self.yhat[:,t]==0)
        #     one_yhat = np.argwhere(self.yhat[:,t]==1)

        #     if np.any(zero_yhat):
        #         self.y[zero_yhat,t] = np.bitwise_or(self.yhat[zero_yhat,t], np.random.binomial(1,self.alpha_rate, size=(len(zero_yhat),1))).astype(np.int)
        #     if np.any(one_yhat):
        #         self.y[one_yhat,t] = np.multiply(self.yhat[one_yhat,t], np.random.binomial(1,1-self.beta_rate, size=(len(one_yhat),1))).astype(np.int)

        #     # true a, what a[t] is fitting
        #     self.ahat[:,t] = np.max(self.s[:,t]*self.w0, axis=1)
            
        #     # compute c[t]
        #     self.c[:,t] = np.log((1-self.alpha_rate2)*(1-self.beta_rate2)/(self.alpha_rate2*self.beta_rate2))*self.y[:,t] - \
        #                 np.log((1-self.alpha_rate2)/self.beta_rate2)

    def model_fit(self, max_iter=200, ds=0.2, thresh=0.5, exact=False):

        max_it = max_iter

        true_neg = []
        false_pos = []
        true_pos = []
        false_neg = []

        it = 0
        timed = []

        N = self.N
        T = self.T

        eta = np.zeros((N,T), dtype='float32')
        nu = np.zeros((N,T,N), dtype='float32')
       
        a = np.zeros((N,T), dtype='float32') 
        w = np.zeros((N,N), dtype='float32') 
        np.fill_diagonal(w,0)

        mu = self.mu 
        sigma = self.sigma

        s = self.s.T.copy().astype(np.float32)   
        c = self.c.astype(np.float32)           
        wh = 0.5*np.ones((N,N), dtype='float32')
        np.fill_diagonal(wh,0)
        mu_m = mu*np.ones((N,N), dtype='float32')
        np.fill_diagonal(mu_m,0)

        a_center = 1-(1/2**np.sum(s,axis=1)).T
        a_cent = np.zeros((N,T))
        for n in range(N):
            a_cent[n] = a_center

        sig_inv = float(1/sigma)

        # cupy
        a = cp.asarray(a, dtype='float32')
        a_cent = cp.asarray(a_cent, dtype='float32')
        c = cp.asarray(c, dtype='float32')
        eta = cp.asarray(eta, dtype='float32')
        w = cp.asarray(w, dtype='float32')
        mu_m = cp.asarray(mu_m, dtype='float32')
        wh = cp.asarray(wh, dtype='float32')
        s = cp.asarray(s, dtype='float32')
        sig_inv = cp.float32(sig_inv)
        step = cp.float32(ds)

        beta1 = cp.float32(0.9)
        beta2 = cp.float32(0.999)

        m_nu = np.zeros((N,T,N), dtype='float32')
        v_nu = np.zeros((N,T,N), dtype='float32')

        m_eta = cp.zeros((N,T), dtype='float32')
        v_eta = cp.zeros((N,T), dtype='float32')

        eps = cp.float32(0.0000001)

        it = 0
        while it < max_it:
            ## using dual decomposition. solve for min a and w, then update eta and nu
    
            print('Iteration ', it)
            timer = time.time()

            # memory load for nu typically too high, so use loop over N neurons
            # and move to gpu incrementally 

            # Solve for min a and w
            for n in range(N): 
                nu_gpu = cp.asarray(nu[n])*s
                if exact:
                    cst = (c[n] - eta[n] + cp.sum(nu_gpu, axis=1))
                    a[n] = 1/(1+cp.exp(-cst))
                    cstw = -mu_m[n] + eta[n].dot(s) - cp.sum(nu_gpu, axis=0)
                    w[n] = 1/(1+cp.exp(-cstw)) 
                else:
                    a[n] = a_cent[n] + sig_inv*(c[n] - eta[n] + cp.sum(nu_gpu, axis=1))
                    w[n] = wh[n] + sig_inv*(-mu_m[n] + eta[n].dot(s) - cp.sum(nu_gpu, axis=0))
            
            # Restrict solutions to the box [0,1]
            a[a<0] = 0
            a[a>1] = 1
            w[w<0] = 0
            w[w>1] = 1

            # Compute updates for dual variable eta
            grad_eta = (a - s.dot(w.T).T)

            # Use Adam for update
            m_eta = beta1*m_eta + (1-beta1)*grad_eta
            v_eta = beta2*v_eta + (1-beta2)*grad_eta**2
            m_hat_eta = m_eta/(1-cp.power(beta1,it+1))
            v_hat_eta = v_eta/(1-cp.power(beta2,it+1))
            eta = eta + step*m_hat_eta / (cp.sqrt(v_hat_eta)+eps)

            # Restrict to the positive orthant
            eta[eta<0] = 0

            # Compute updates for dual variable nu
            for n in range(N):
                grad_nu = (w[n,None,:]-a[n,:,None])*s

                # Move to GPU
                nu_gpu = cp.asarray(nu[n])
                m_nu_gpu = cp.asarray(m_nu[n])
                v_nu_gpu = cp.asarray(v_nu[n])
                m_nu_gpu = beta1*m_nu_gpu + (1-beta1)*grad_nu
                v_nu_gpu = beta2*v_nu_gpu + (1-beta2)*grad_nu**2
                m_hat_nu = m_nu_gpu/(1-cp.power(beta1,it+1))
                v_hat_nu = v_nu_gpu/(1-cp.power(beta2,it+1))
                nu_gpu = nu_gpu + step*m_hat_nu / (cp.sqrt(v_hat_nu)+eps)

                # Restrict to positive orthant
                nu_gpu[nu_gpu<0] = 0

                # Move back to CPU
                nu[n] = cp.asnumpy(nu_gpu)
                m_nu[n] = cp.asnumpy(m_nu_gpu)
                v_nu[n] = cp.asnumpy(v_nu_gpu)

            # End iteration
            print('Time :', time.time()-timer)
            timed.append(time.time()-timer)
            it += 1
            
            # Compute TN, etc, using ground truth w0
            tp = w[np.nonzero(self.w0)]
            tp[tp<thresh] = 0
            fn = len(tp) - np.count_nonzero(tp)
            false_neg.append([fn])
            tp = np.count_nonzero(tp)
            true_pos.append([tp])

            tn = w[np.nonzero(self.w0-1)]
            tn[tn<thresh] = 0
            fp = np.count_nonzero(tn)
            false_pos.append([fp])
            tn = len(tn) - np.count_nonzero(tn)
            true_neg.append([tn])

        # End of model fit
        print('--------- Average time per iteration: ', np.mean(np.array(timed)))

        # Move to CPU to return solutions
        a = cp.asnumpy(a)
        w = cp.asnumpy(w)

        return a, w, true_neg, false_pos, true_pos, false_neg

    def data_gen_stream(self, window=10):
        ''' Only generate data for the initial window requested
        ''' 
        self.window = window
        N = self.N

        self.c = np.zeros((N,window), dtype=np.float32)
        self.s = np.zeros((N,window), dtype='int')
        self.y = np.zeros((N,window), dtype='int')
        self.ahat = np.zeros((N,window))
        self.yhat = np.zeros((N,window), dtype='int')
        self.s = np.random.binomial(1, (self.N_S/self.N), size=((self.N,self.window)))

        self.yhat = np.max(np.multiply(self.s[np.newaxis,:,:],self.w0[...,np.newaxis]).astype(np.int), axis=1)
        zero_yhat = self.yhat == 0

        alpha_error = np.random.binomial(1,self.alpha_rate, size=(self.yhat[zero_yhat].shape))
        self.y[zero_yhat] = np.bitwise_or(self.yhat[zero_yhat], alpha_error).astype(np.int)
        beta_error = np.random.binomial(1,1-self.beta_rate, size=(self.yhat[~zero_yhat].shape))
        self.y[~zero_yhat] = np.multiply(self.yhat[~zero_yhat], beta_error).astype(np.int)

        self.ahat = self.yhat

        self.c = (np.log((1-self.alpha_rate2)*(1-self.beta_rate2)/(self.alpha_rate2*self.beta_rate2))*self.y - \
                        np.log((1-self.alpha_rate2)/self.beta_rate2)).astype('float32')

    def model_stream(self, iter_step=5, ds=0.2, thresh=0.5, adaptive=False, largeN=False):
        window = self.window

        true_neg = []
        false_pos = []
        true_pos = []
        false_neg = []

        tstep = 1 
    
        N = self.N

        eta = np.zeros((N,window))
        nu = np.zeros((N,window,N))

        a = np.zeros((N,window))
        w = np.zeros((N,N))

        mu = self.mu
        sigma = self.sigma

        s = self.s.T.copy()    
        c = self.c            
        wh = 0.5*np.ones((N,N))
        np.fill_diagonal(wh,0)
        mu_m = mu*np.ones((N,N))
        np.fill_diagonal(mu_m,0)
        a_cent = np.zeros((N,window))

        ## cupy
        a = cp.asarray(a, dtype='float32')
        a_cent = cp.asarray(a_cent, dtype='float32')
        c = cp.asarray(c.copy(), dtype='float32')
        eta = cp.asarray(eta, dtype='float32')
        w = cp.asarray(w, dtype='float32')
        mu_m = cp.asarray(mu_m, dtype='float32')
        wh = cp.asarray(wh, dtype='float32')
        s = cp.asarray(s.copy(), dtype='float32')
        sig_inv = cp.float32(1/sigma)
        
        # restrict initial data to the window size
        s = s[:window,:]
        c = c[:,:window]

        step = ds 
        step = cp.float32(step)

        # large n requires keeping nu on the CPU in a list
        if largeN:
            ## larger N
            nu_sum = cp.zeros((N,N))
            nu_sum_a = cp.zeros((N,window))
            for n in range(N):
                nu_gpu = cp.asarray(nu[n,:window,:])
                nu_sum[n] = cp.sum(nu_gpu*s[:window,:], axis=0)
                nu_sum_a[n] = cp.sum(nu_gpu*s[:window,:], axis=1)
        else:
            ## smaller N
            nu = cp.asarray(nu, dtype='float32')
            nu_sum = cp.sum(nu[:,:window,:]*s[:window, :], axis=1)

        # do 10 neurons at a time; empirically found to be fastest
        # on the development machine
        stepn = 10

        # streaming starts after a window of tests 
        for t in range(window, self.T+1, tstep):

            timer = time.time()

            a_center = 1-(1/2**cp.sum(s,axis=1)).T
            for n in range(N):
                a_cent[n] = a_center

            for i in range(iter_step):
                
                if largeN:
                    a = a_cent + sig_inv*(c - eta + nu_sum_a)
                else:
                    a = a_cent + sig_inv*(c - eta + cp.sum(nu*s, axis=2))
                a[a<0] = 0
                a[a>1] = 1

                w = wh + sig_inv*(-mu_m + eta.dot(s) - nu_sum)
                w[w<0] = 0
                w[w>1] = 1

                grad_eta = a - s.dot(w.T).T
                eta = eta + step*(grad_eta)
                eta[eta<0] = 0

                if largeN:
                    for n in range(0,N,stepn):
                        nu_gpu = cp.asarray(nu[n:n+stepn])
                        grad_nu = (w[n:n+stepn,None,:]-a[n:n+stepn,:,None])*s
                        nu_gpu = nu_gpu + step*(grad_nu)
                        nu_gpu[nu_gpu<0] = 0
                        nu_sum_a[n:n+stepn] = cp.sum(nu_gpu*s, axis=2) ##CHANGE IF 2D NOT USING STEPN
                        nu[n:n+stepn] = cp.asnumpy(nu_gpu)
                else:
                    grad_nu = (w[:,None,:]-a[:,:,None])*s
                    nu = nu + step*(grad_nu)
                    nu[nu<0] = 0

                step /=  1.001
                step = cp.float32(step)

            eta = cp.roll(eta,-1,axis=1)
            eta[:,-1] = 0

            if largeN:
                for n in range(0,N,stepn):
                    nu_gpu = cp.asarray(nu[n:n+stepn])
                    nu_sum[n:n+stepn] += nu_gpu[:,-1,:]*s[-1,:] ## CHANGE IF 2D
                    nu_gpu = cp.roll(nu_gpu,-1,axis=1)
                    nu_gpu[:,-1,:] = 0
                    nu[n:n+stepn] = cp.asnumpy(nu_gpu)
            else:
                nu_sum += nu[:,-1,:]*s[-1,:]
                nu = cp.roll(nu,-1,axis=1)
                nu[:,-1,:] = 0
         
            # Choose next targets based on entropy or random (default)
            if t<self.T-1: 
                s = cp.roll(s,-1,axis=0)
                if adaptive:
                    h = optimH(N, w)
                    ent = cp.sum(h,axis=0)
                    ent_sort = cp.argsort(ent)
                    new_choice = ent_sort[-self.N_S:]
                    s[-1,:] = 0
                    s[-1, new_choice] = 1
                else:
                    new_choice = cp.random.binomial(1, (self.N_S/self.N), size=((self.N,1)))
                    s[-1] = cp.squeeze(new_choice)

                print('Test ', t, ' time: ', time.time()-timer)
            
                ## compute deterministic yhat
                y = np.zeros(N)
                yhat = np.max(np.multiply(cp.asnumpy(s[-1,:].T), self.w0).astype(np.int), axis=1)

                zero_yhat = np.argwhere(yhat==0)
                one_yhat = np.argwhere(yhat==1)

                if np.any(zero_yhat):
                    y[zero_yhat] = np.bitwise_or(yhat[zero_yhat], np.random.binomial(1,self.alpha_rate, size=(len(zero_yhat),1))).astype(np.int)
                if np.any(one_yhat):
                    y[one_yhat] = np.multiply(yhat[one_yhat], np.random.binomial(1,1-self.beta_rate, size=(len(one_yhat),1))).astype(np.int)

                # compute c[t]
                c_new = np.log((1-self.alpha_rate2)*(1-self.beta_rate2)/(self.alpha_rate2*self.beta_rate2))*y - \
                            np.log((1-self.alpha_rate2)/self.beta_rate2)

                c = cp.roll(c,-1,axis=1)
                c[:,-1] = 0
                c[:,-1] = cp.asarray(c_new)


            tp = w[np.nonzero(self.w0)]
            tp[tp<thresh] = 0
            fn = len(tp) - np.count_nonzero(tp)
            false_neg.append([fn])
            tp = np.count_nonzero(tp)
            true_pos.append([tp])

            tn = w[np.nonzero(self.w0-1)]
            tn[tn<thresh] = 0
            fp = np.count_nonzero(tn)
            false_pos.append([fp])
            tn = len(tn) - np.count_nonzero(tn)
            true_neg.append([tn])

        return cp.asnumpy(a), cp.asnumpy(w), true_neg, false_pos, true_pos, false_neg

    def naive(self, bayesian=False, thresh=0.5):
        ''' Includes datagen since stim is different
            Stim each neuron one at a time, for N_S stim,
            average the results e.g. (4/5=0.8) confidence of weight
        '''

        self.c = np.zeros((N,T), dtype=np.float32)
        self.s = np.zeros((N,T), dtype='int')
        self.y = np.zeros((N,T), dtype='int')
        self.yhat = np.zeros((N,T), dtype='int')

        true_neg = []
        false_pos = []
        true_pos = []
        false_neg = []
        average_res = np.zeros((N,N)) + 0.5
        beta_map_res = np.zeros((N,N))
        cumm = np.zeros((N,N))
        n_counter = np.zeros((N)) #how often was this neuron stimulated and others seen

        for t in range(self.T):
            print('Test ', t)

            stim = random.sample(range(0, self.N), 1)
            self.s[stim,t] = 1 

            n_counter[stim] += 1

            self.yhat[:, t] = np.max(np.multiply(self.s[:,t], self.w0).astype(np.int), axis=1) 

            zero_yhat = np.argwhere(self.yhat[:,t]==0)
            one_yhat = np.argwhere(self.yhat[:,t]==1)

            if np.any(zero_yhat):
                self.y[zero_yhat,t] = np.bitwise_or(self.yhat[zero_yhat,t], np.random.binomial(1,self.alpha_rate, size=(len(zero_yhat),1))).astype(np.int)
            if np.any(one_yhat):
                self.y[one_yhat,t] = np.multiply(self.yhat[one_yhat,t], np.random.binomial(1,1-self.beta_rate, size=(len(one_yhat),1))).astype(np.int)
            
            # compute either the average result and the result using bayesian inference
            if bayesian:
                stimmed = np.squeeze(np.argwhere(self.s[stim,:]==1))  # all t where this neuron was stimmed
                if stimmed.ndim > 1:
                    stimmed = stimmed[:,1]
                    num_ones = np.sum(self.y[:,stimmed], axis=1)[...,np.newaxis]
                else:
                    stimmed = stimmed[1]
                    num_ones = self.y[:,stimmed][...,np.newaxis]
                # using Beta(1,5) prior, change here to try other priors
                beta_map_res[:,stim] = (num_ones + 1 - 1) / (n_counter[stim]+5+1 - 2)
                used_res = beta_map_res
            else:
                average_res[:,stim] = (self.y[:,t][...,np.newaxis] + (n_counter[stim]-1)*average_res[:,stim])/n_counter[stim]
                used_res = average_res

            cumm[:,stim] += self.y[:,t][...,np.newaxis]

            tp = used_res[np.nonzero(self.w0)]
            tp[tp<thresh] = 0
            fn = len(tp) - np.count_nonzero(tp)
            false_neg.append([fn])
            tp = np.count_nonzero(tp)
            true_pos.append([tp])

            tn = used_res[np.nonzero(self.w0-1)]
            tn[tn<thresh] = 0
            fp = np.count_nonzero(tn)
            false_pos.append([fp])
            tn = len(tn) - np.count_nonzero(tn)
            true_neg.append([tn])

        return used_res, n_counter, cumm, true_neg, false_pos, true_pos, false_neg

def conservH(w):
    h = -w*np.log(w+np.finfo(float).eps) - (1-w)*np.log(1-w+np.finfo(float).eps)
    H = np.sum(h)
    return h, H

def optimH(N, w):
    ''' Entropy calculation for choosing stimuli
    ''' 
    h = cp.log(2) - (2*w-1)**2
    return h
    
#############################################

if __name__=="__main__":
    
    # base case parameters
    T = 1000
    N = 1000
    N_S = 10
    alpha = 0.05        # what it actually is
    beta = 0.05
    alpha2 = 0.05       # what we assume it is
    beta2 = 0.05
    sparsity = ceil(N**0.3)/N 
    mu = 0
    sigma = 0.1
    thresh = 0.5

    timerfull = time.time()

    ## pick a seed
    np.random.seed(100)
    random.seed(100)

    ## initialize model
    MS = model_stim(T, N, N_S, alpha=alpha, beta=beta, alpha2=alpha2, beta2=beta2, sparsity=sparsity, mu=mu, sigma=sigma)
    
    if len(sys.argv)<=1:
        print('Need to choose a method for fitting.')
        print('Base options are: batch, stream, adapt, naive, exact.')
        raise SystemExit

    method = sys.argv[1]

    if method == 'batch':
        ## run batch model fitting
        max_iter = 50
        ds = 0.01
        MS.data_gen()
        a, w, tn, fp, tp, fn = MS.model_fit(max_iter=max_iter, ds=ds, thresh=thresh)

    elif method == 'stream' or method == 'adapt':
        ## run streaming / adaptive model fitting
        window = 10
        max_iter = 5
        ds = 0.1
        adaptive = False if method == 'stream' else True
        MS.data_gen_stream(window=window)
        a, w, tn, fp, tp, fn = MS.model_stream(iter_step=max_iter, ds=ds, thresh=thresh, adaptive=adaptive)
    
    elif method == 'naive':
        ## run naive model fitting
        ## Note naive does its own data generation
        use_bayesian = False
        w, count, cumm, tn, fp, tp, fn = MS.naive(bayesian=use_bayesian, thresh=thresh)

    elif method == 'exact':
        ## run exact Bayesian inference
        ## this uses a different 'base case'
        N = 200
        T = 1000
        alpha = beta = alpha2 = beta2 = 0.02
        mu = 4
        exactMS = model_stim(T, N, N_S, alpha=alpha, beta=beta, alpha2=alpha2, beta2=beta2, sparsity=sparsity, mu=mu, sigma=sigma)
        exactMS.data_gen()
        max_iter = 200
        ds = 0.05
        a, w, tn, fp, tp, fn = exactMS.model_fit(max_iter=max_iter, ds=ds, thresh=thresh, exact=True)

    elif method == 'batch_T':
        ## run a set of tests for batch mode and plot the results
        tests = np.array([10,20,50,100,200,500,800,1000])
        tn = []
        tp = []
        fn = []
        fp = []
        max_iter = 50
        ds = 0.01
        for T in tests:
            MS = model_stim(T, N, N_S, alpha=alpha, beta=beta, alpha2=alpha2, beta2=beta2, sparsity=sparsity, mu=mu, sigma=sigma)
            MS.data_gen()
            a, w, tn_T, fp_T, tp_T, fn_T = MS.model_fit(max_iter=max_iter, ds=ds, thresh=thresh)
            tn.append(tn_T[-1])
            tp.append(tp_T[-1])
            fn.append(fn_T[-1])
            fp.append(fp_T[-1])
            print('Done with test size ', T)
        tn = np.array(tn)
        fp = np.array(fp)
        tp = np.array(tp)
        fn = np.array(fn)
        plt.figure(1)
        plt.plot(tests, tn/(tn+fp), label='Specificity')
        plt.plot(tests, tp/(tp+fn), label='Sensitivity')
        plt.legend(loc='lower right')
        plt.show()

    else:
        print('Chosen method not in list of options.')
        print('Base options are: batch, stream, adapt, naive, exact.')
        raise SystemExit

    ## end results
    tn = np.array(tn)
    fp = np.array(fp)
    tp = np.array(tp)
    fn = np.array(fn)

    print('End sensitivity, specificity: ', tp[-1]/(tp[-1]+fn[-1]), tn[-1]/(tn[-1]+fp[-1]))
    
    print('TOTAL TIME ', time.time()-timerfull)
    
    ########## Uncomment below for saving and other plotting example

    # # Save results
    # np.savetxt('./output/sens_fit'+str(method)+'.txt', tp/(tp+fn))
    # np.savetxt('./output/sens_fit'+str(method)+'.txt', tn/(tn+fp))
    
    # # Can also look at the histogram of inferred connections
    # plt.figure()
    # plt.hist(w, bins=20)
    # plt.show()