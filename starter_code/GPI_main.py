import numpy as np 
import copy
from scipy.stats import multivariate_normal
from tqdm import tqdm
import os

class GPI():
    def __init__(self,):
        # Discretization 
        self.nx = 20
        self.ny = 20
        self.nth = 20
        self.nt = 100
        self.ts = 0.5
        self.nv = 10
        self.nw = 20

        self.M = self.nx*self.ny*self.nth*self.nt
        self.N = self.nv*self.nw
        
        #State Limits
        self.pmax = np.array([3,3])
        self.pmin = np.array([-3,-3])
        self.vlim = np.array([0,1])
        self.wlim = np.array([-1,1])
        
        #Cost and noise parameters
        self.Q = np.eye(2)
        self.R = np.eye(2)
        self.q = 1
        cov = np.zeros([3,3])
        cov[0,0], cov[1,1], cov[2,2] = 0.4,0.4,0.04
        self.cov = cov
        
        #Run the code
        
        self.tuple_state = self.build_state_and_control_space()
        self.statespace = self.tuple_state[0]
        self.controlspace = self.tuple_state[1]
        if(os.path.exists('stagecost.npy')):
            self.stagecost = np.load('stagecost.npy')
        else:
            self.stagecost = self.stage_cost()
        self.tuple_prob = self.transition_probabilities()
        self.transition_probs = self.tuple_prob[0]
        self.new_indices = self.tuple_prob[1]
        self.policy = self.policy_iteration()
        
    def build_state_and_control_space(self):
        err_x = np.linspace(self.pmin[0],self.pmax[0],self.nx + 1)
        err_y = np.linspace(self.pmin[1],self.pmax[1],self.ny + 1)
        err_th = np.linspace(-np.pi,np.pi,self.nth + 1)
        
        M = self.M
        state = np.zeros([4,M])
        
        a = 0
        print('StateSpace')
        for t in tqdm(range(self.nt)):
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nth):
                        state[:,a] = np.array([t,err_x[i],err_y[j],err_th[k]])
                        a += 1
                        
        velocity = np.linspace(self.vlim[0],self.vlim[1],self.nv +1)
        omega = np.linspace(self.wlim[0],self.wlim[1],self.nw + 1)
        
        control = np.zeros([2,self.N])
        b = 0
        for p in range(self.nv):
            for q in range(self.nw):
                control[:,b] = np.array([velocity[p],omega[q]])
                b += 1
        
        return state,control
    
    def isValid(self,state):
        if(np.any(state[1:3]< self.pmin) or np.any(state[1:3] > self.pmax)):
            return False
        elif(np.linalg.norm(state - np.array([-2,-2])) < 0.5 or np.linalg.norm(state - np.array([1,2])) < 0.5):
            return False
    
    def stage_cost(self):
        M = self.M
        N = self.N
        L = np.zeros([M,N])
        if(self.statespace is None):
            self.statespace, self.controlspace = self.build_state_and_control_space()
        else:
            state,u = self.statespace,self.controlspace
        p_ = state[1:3,:]
        th_ = state[3,:]
        print(p_[:,np.newaxis,5].T)
        print('StageCost')
        for i in tqdm(range(M)):
            for j in range(N):
                if not self.isValid(p_[:,i]):
                    L[i,j] = np.inf
                L[i,j] =  p_[:,np.newaxis,i].T @ self.Q @ p_[:,np.newaxis,i] + u[:,np.newaxis,j].T @ self.R @ u[:,np.newaxis,j] + self.q * (1 - np.cos(th_[i]))**2
        np.save('stagecost.npy',L)
        return L
        
    def motion_model(self,state_idx,action_idx,t):
        ref = lissajous(t)
        ref_next = lissajous(t+1)
        # print(self.statespace[:,0], self.statespace[1:3,state_idx], self.statespace[3,state_idx])
        err = self.statespace[1:,state_idx]
        th = err[2]
        u = self.controlspace[:,action_idx]
        # print(np.shape(p),np.shape(th))
        p_ = err.reshape([3,1])
        u = u.reshape([2,1])
        
        next_state = p_ + self.ts * np.array([[np.cos(th + ref[2]),0],[np.sin(th + ref[2]),0],[0,1]]) @ u \
            + np.vstack((ref[0] - ref_next[0],ref[1] - ref_next[1],ref[2] - ref_next[2]))
        
        # State Limits
        if(np.any(next_state[0:2]>self.pmax)):
            self.stagecost[state_idx,action_idx] = np.inf
            next_state[0:2,0] = self.pmax
        elif(np.any(next_state[0:2]<self.pmin)):
            self.stagecost[state_idx,action_idx] = np.inf
            next_state[0:2,0] = self.pmin

        # Collision avoidance
        if(np.linalg.norm(next_state[0:2] - np.array([-2,-2])) < 0.5 or np.linalg.norm(next_state[0:2] - np.array([1,2])) < 0.5):
            self.stagecost[state_idx,action_idx] = np.inf 
        
        # Angle wrapping
        n= int(np.abs((next_state[2]/np.pi)))
        if(n%2==0):
            next_state[2,0]= next_state[2,0] -(n*(np.pi)*np.sign(next_state[2]))
        else:
            next_state[2,0]= next_state[2,0] -(n+1)*np.pi*(np.sign(next_state[2]))

        return next_state
    
    def transition_probabilities(self):
        M = self.M 
        N = self.N
        st_t = self.nx * self.ny * self.nth
        P_transition = np.zeros([M,N,st_t]) # 9 connected cells
        state,u = self.statespace, self.controlspace
        print('Transition Probabilities')
        for m in tqdm(range(M)):
            t = state[0,m]
            ref = lissajous(t)
            for n in range(N):
                # next_state_mean = self.motion_model(state[:,m],u[:,n],t) 
                next_state_mean = self.motion_model(m,n,t) 
                if(t+1 % 100 == 0):
                    new_state_idx = np.arange(0,st_t,dtype=int)
                else:
                    new_state_idx = np.arange(np.floor(m/st_t)*st_t + st_t, np.floor(m/st_t)*st_t + 2*st_t,dtype=int)
                
                p_trans = np.zeros(st_t)
                p_trans = multivariate_normal.pdf(state[1:,new_state_idx].T,next_state_mean.flatten(),self.cov)
                p_trans /= np.sum(p_trans)
                P_transition[m,n,:] = p_trans
        return P_transition, new_state_idx
    
    def policy_iteration(self):
        P,next_state_idx = self.transition_probabilites()
        V = np.zeros(self.M)
        pi = np.zeros(self.M) 
        if(self.stagecost is None):
            L = self.stage_cost()
        else:
            L = self.stagecost

        for x in tqdm(range(100)):
            V_ = copy.deepcopy(V)
            Q = L + np.sum(P * V_[next_state_idx],axis = 2)
            V = np.min(Q,axis =1)
            pi = np.argmin(Q,axis=1)
            if np.linalg.norm(V- V_) <= 0.1:
                break
        return pi


time_step = 0.5
def lissajous(k):
    xref_start = 0
    yref_start = 0
    A = 2
    B = 2
    a = 2*np.pi/50
    b = 3*a
    T = np.round(2*np.pi/(a*time_step))
    k = k % T
    delta = np.pi/2
    xref = xref_start + A*np.sin(a*k*time_step + delta)
    yref = yref_start + B*np.sin(b*k*time_step)
    v = [A*a*np.cos(a*k*time_step + delta), B*b*np.cos(b*k*time_step)]
    thetaref = np.arctan2(v[1], v[0])
    return [xref, yref, thetaref]

if __name__ == '__main__':
    policy = GPI()
    np.save('policy.npy',policy)