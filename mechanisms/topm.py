import math
import numpy as np

class TOPM():
    def __init__(self, d, eps, rng, range=1.0, k=1, t=None):
        self.d = d
        self.eps = eps
        self.rng = rng
        self.range = range
        # eps_star for TO_single
        self.k = k
        self.eps_k = self.eps / self.k
        if t is None:
            self.t = np.exp(self.eps_k / 3)
        else:
            self.t = t
        self.P0_0, self.C = self.get_Ps()

        self.Px_0, self.Px_1 = self.construct_prob()

        # for piecewise mech
        self.A = (np.exp(self.eps_k) + self.t) * (self.t + 1) / (self.t * (np.exp(self.eps_k)-1))

        # HM
        self.beta = self.get_beta()
        if self.eps_k < np.log(2):
            self.x_star = 0
        else:
            self.x_star = (self.beta - 1) * self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2 / ( 2 * (np.exp(self.eps_k) - self.P0_0)**2 * (self.beta * (np.exp(self.eps_k) + self.t) - np.exp(self.eps_k) + 1))
        # self.x_star = (self.beta - 1) * self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2 / ( 2 * (np.exp(self.eps_k) - self.P0_0)**2 * (self.beta * (np.exp(self.eps_k) + self.t) - np.exp(self.eps_k) + 1))

    def construct_prob(self):
        Px_1 = (np.exp(self.eps_k) - self.P0_0) / (np.exp(self.eps_k) + 1) - (1 - self.P0_0) / 2
        Px_0 = self.P0_0 / np.exp(self.eps_k) - self.P0_0

        return Px_0, Px_1

    def get_Ps(self):
        if self.eps_k < np.log(2):
            P0_0 = 0
        elif self.eps_k <= np.log((3 + np.sqrt(65)) / 2):
            delta_0 = np.exp(4*self.eps_k) + 14*np.exp(3*self.eps_k) + 50*np.exp(2*self.eps_k) - 2 * np.exp(self.eps_k) + 25
            delta_1 = -2*np.exp(6*self.eps_k) -42*np.exp(5*self.eps_k) -270*np.exp(4*self.eps_k) -404*np.exp(3*self.eps_k) -918*np.exp(2*self.eps_k) +30*np.exp(self.eps_k) -250
            
            P0_0 = (-1/6) * ( -np.exp(2*self.eps_k) - 4*np.exp(self.eps_k) - 5 + 2 * np.sqrt(delta_0) * np.cos(np.pi/3 + (1/3) * np.arccos(-delta_1 / (2*np.sqrt(delta_0**3))) ))
        else:
            P0_0 = np.exp(self.eps_k) / (np.exp(self.eps_k) + 2)

        C = (np.exp(self.eps_k) + 1) / ((np.exp(self.eps_k)-1) * (1 - P0_0 / np.exp(self.eps_k)))

        return P0_0, C

    def get_beta(self):
        if self.eps_k > 0 and self.eps_k < 0.610986:
            beta = 0
        elif self.eps_k < np.log(2):
            beta = (2*(np.exp(self.eps_k) - self.P0_0)**2 * (np.exp(self.eps_k) -1) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2) / (2 * (np.exp(self.eps_k)-self.P0_0)**2 * (np.exp(self.eps_k) + self.t) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2)
        else:
            c = np.exp(self.eps_k)
            A = self.P0_0**2 * c**2 * (c+1)**4 / (4 * (c + self.t)**2 * (c - self.P0_0)**4 * (c-1)) - self.P0_0**2 * c**2 * (c+1)**4 / (2 * (c+self.t)**2 * (c-1) * (c-self.P0_0)**4) + ( (self.t+1)**3 + c - 1) / (3 * self.t**2 * (c - 1)**2) - (1 - self.P0_0) * c**2 * (c+1)**2 / ((c+self.t)*(c-1)**2 * (c-self.P0_0)**2)
            B = (-(1 + self.t)**2 * self.P0_0**2 * c**2 * (c+1)**4) / (4* (c+self.t)**2 * (c-self.P0_0)**4 * (c-1))

            beta1 = (2*(np.exp(self.eps_k) - self.P0_0)**2 * (np.exp(self.eps_k) -1) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2) / (2 * (np.exp(self.eps_k)-self.P0_0)**2 * (np.exp(self.eps_k) + self.t) - self.P0_0 * np.exp(self.eps_k) * (np.exp(self.eps_k) + 1)**2)

            beta = (-np.sqrt(B/A) + np.exp(self.eps_k) -1) / (np.exp(self.eps_k) + np.exp(self.eps_k/3))
        
        return beta

    def TO_single(self, x):
        x = x / self.range
        P_0_x = self.P0_0 + self.Px_0 * np.abs(x)

        if x >= 0 and x <= 1:
            P_C_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(x)
            P_mC_x = 1 - P_C_x - P_0_x
        else:
            P_mC_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(x)
            P_C_x = 1 - P_0_x - P_mC_x

        p = [P_mC_x, P_0_x, P_C_x]
        p = np.asarray(p).astype('float64')
        p /= p.sum()
        u = self.rng.choice([-1, 0, 1], 1, p = p)

        return u * self.C * self.range
    
    def TO_batch(self, data):
        data = data.reshape(-1) / self.range
        noisy_output = np.zeros_like(data)
        
        minus_idx = np.argwhere(data < 0).reshape(-1)
        P_0_x = self.P0_0 + self.Px_0 * np.abs(data[minus_idx])
        P_mC_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(data[minus_idx])

        minus_random = self.rng.random(len(minus_idx))
        mC_idx = np.argwhere(minus_random < P_mC_x).reshape(-1)
        zero_idx = np.argwhere((minus_random >= P_mC_x) & (minus_random < (P_mC_x + P_0_x)) ).reshape(-1)
        C_idx = np.argwhere(minus_random >= (P_mC_x + P_0_x)).reshape(-1)
        noisy_output[minus_idx[mC_idx]] = -self.C
        noisy_output[minus_idx[zero_idx]] = 0
        noisy_output[minus_idx[C_idx]] = self.C

        plus_idx = np.argwhere(data >= 0).reshape(-1)
        P_0_x = self.P0_0 + self.Px_0 * np.abs(data[plus_idx])
        P_C_x = (1 - self.P0_0) / 2 + self.Px_1 * np.abs(data[plus_idx])

        plus_random = self.rng.random(len(plus_idx))
        C_idx = np.argwhere(plus_random < P_C_x).reshape(-1)
        zero_idx = np.argwhere((plus_random >= P_C_x) & (plus_random < (P_C_x + P_0_x)) ).reshape(-1)
        mC_idx = np.argwhere(plus_random >= (P_C_x + P_0_x)).reshape(-1)
        noisy_output[plus_idx[mC_idx]] = -self.C
        noisy_output[plus_idx[zero_idx]] = 0
        noisy_output[plus_idx[C_idx]] = self.C

        return noisy_output * self.range, None
    
    def TO_max_var(self):
        b = self.P0_0 * (1 - 1/np.exp(self.eps_k))

        if self.C**2*b < 2:
            print(f'max is :{self.C / 2}')
            V = (1 - self.P0_0) * self.C**2 + self.C**4 * b**2 / 4
        else:
            print(f'max is 1')
            V = (1 - self.P0_0 + b) * self.C**2 - 1

        # if self.eps_k < np.log(2):
        #     return (np.exp(self.eps_k) + 1)**2 / (np.exp(self.eps_k) - 1)**2
        # elif self.eps_k <= np.log(5.53):
        #     return np.exp(2*self.eps_k) * (np.exp(self.eps_k) + 1)**2 / (np.exp(self.eps_k) - 1)**2 * ( (1-self.P0_0) / (np.exp(self.eps_k) - self.P0_0)**2 + (np.exp(self.eps_k) + 1)**2 * self.P0_0**2 / ( 4 * (np.exp(self.eps_k - self.P0_0))**4) )
        # else:
        #     return (np.exp(self.eps_k) + 2) * (np.exp(self.eps_k) + 10) / ( 4 * (np.exp(self.eps_k) -1)**2 )

        # if self.eps_k < np.log(2):
        #     V = (np.exp(self.eps_k) + 1)**2 / (np.exp(self.eps_k) - 1)**2
        # elif self.eps_k < np.log((3 + np.sqrt(65))/ 2):
        #     V = ((np.exp(self.eps_k) + 1)**2 * np.exp(self.eps_k * 2) / (np.exp(self.eps_k)-1)**2) * ( (1-self.P0_0) / (np.exp(self.eps_k) - self.P0_0)**2 + (np.exp(self.eps_k)+1)**2 * self.P0_0**2 / (4 * (np.exp(self.eps_k) - self.P0_0)**4)  )
        # else:
        #     print('aweaweqaweawe')
        #     V = (np.exp(self.eps_k) + 2) * (np.exp(self.eps_k) + 10) / (4 * np.exp(self.eps_k) - 1)**2

        return V
    
    def TO_multi(self, x):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)

        # if self.topk:
        #     js = np.argpartition(np.abs(x), -self.k)[-self.k:]

        # else:
        js = self.rng.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            y[j] = self.d * self.TO_single(x[j]) / self.k

        return y.reshape(original_shape)
    
    # def TO_multi_batch(self, data):
    #     sampled_dims = np.random.choice(len(data), self.k, replace=False)
    #     noisy_output = np.zeros_like(data)

    #     noisy_output[sampled_dims], _ = self.TO_batch(data[sampled_dims])
    #     # no_inds = inds[no_pm_inds[0]]
    #     # pm_inds = inds[no_pm_inds[1]]
    #     return noisy_output * self.d / self.k

    def TO_multi_batch(self, sampled_values, row_indices, sampled_col_indices, num_sam, d, k):
        perturbed_values, _ = self.TO_batch(sampled_values.flatten())
        perturbed_values = perturbed_values.reshape(num_sam, k)

        # 4. 결과 배열을 생성하고, 교란된 값들을 원래 위치에 한 번에 삽입합니다.
        noisy_data = np.zeros((num_sam, d))
        noisy_data[row_indices, sampled_col_indices] = perturbed_values
        
        # 5. 통계적 보정을 위해 d/k 스케일링을 적용합니다.
        return noisy_data * (d / k)
    
    def TO_multi_simul(self, x, m=None):
        prob = self.k / self.d
        if np.random.rand() < prob:
            y = self.TO_single(x)
        else:
            y = 0

        return y * self.d / self.k
    
    def TO_multi_simul_batch(self, data):
        data = data.reshape(-1)
        prob = self.k / self.d
        inds = np.argwhere(self.rng.random(len(data)) < prob).reshape(-1)
        noisy_output = np.zeros_like(data)

        noisy_output[inds], _ = self.TO_batch(data[inds])

        return noisy_output * self.d / self.k, None
    
    def TO_var(self, x):
        b = self.P0_0*(1 - 1/np.exp(self.eps_k))
        C = (np.exp(self.eps_k) + 1) / ((np.exp(self.eps_k) -1) * (1 - self.P0_0 / np.exp(self.eps_k)))

        return C**2 * (1 - self.P0_0) + C**2 * b * np.abs(x) - x**2
    
    def PM_single(self, x):
        u = self.rng.random()
        x = x / self.range

        l = (np.exp(self.eps_k) + self.t) * (x * self.t - 1) / (self.t * (np.exp(self.eps_k) - 1))
        r = (np.exp(self.eps_k) + self.t) * (x * self.t + 1) / (self.t * (np.exp(self.eps_k) - 1))

        if u < np.exp(self.eps_k) / (self.t + np.exp(self.eps_k)):
            y = self.rng.uniform(l, r)
        else:
            length_l = abs(l - (-self.A))
            legnth_r = abs(self.A - r)
            interval_l = length_l / (length_l + legnth_r)
            
            if self.rng.random() < interval_l:
                y = self.rng.uniform(-self.A, l)
            else:
                y = self.rng.uniform(r, self.A)
            # dist_ratio = (l + self.A) / (self.A - r)
            # if self.rng.random() > 1 / (1 + dist_ratio):
            #     y = self.rng.uniform(-self.A, l)
            # else:
            #     y = self.rng.uniform(r, self.A)
        
        return y * self.range
    
    def PM_multi(self, x, m=None):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)

        # if self.topk:
        #     js = np.argpartition(np.abs(x), -self.k)[-self.k:]

        # else:
        js = self.rng.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])

        for j in js:
            noisy_output = self.PM_single(x[j])
            if m is not None:
                import utils
                noisy_output = utils.discretization(self.A, noisy_output / self.range, self.rng, m) * self.range
            y[j] = self.d * noisy_output / self.k

        return y.reshape(original_shape)
    
    # def PM_multi_batch(self, data):
    #     sampled_dims = np.random.choice(len(data), self.k, replace=False)
    #     noisy_output = np.zeros_like(data)

    #     noisy_output[sampled_dims], _ = self.PM_batch(data[sampled_dims])
    #     # no_inds = inds[no_pm_inds[0]]
    #     # pm_inds = inds[no_pm_inds[1]]
    #     return noisy_output * self.d / self.k

    def PM_multi_batch(self, sampled_values, row_indices, sampled_col_indices, num_sam, d, k):
        perturbed_values, _ = self.PM_batch(sampled_values.flatten())
        perturbed_values = perturbed_values.reshape(num_sam, k)

        # 4. 결과 배열을 생성하고, 교란된 값들을 원래 위치에 한 번에 삽입합니다.
        noisy_data = np.zeros((num_sam, d))
        noisy_data[row_indices, sampled_col_indices] = perturbed_values
        
        # 5. 통계적 보정을 위해 d/k 스케일링을 적용합니다.
        return noisy_data * (d / k)
    
    def PM_multi_simul(self, x, m=None):
        prob = self.k / self.d
        if np.random.rand() < prob:
            y = self.PM_single(x)
        else:
            y = 0

        return y * self.d / self.k
    
    def PM_multi_simul_batch(self, data):
        data = data.reshape(-1)
        prob = self.k / self.d
        inds = np.argwhere(self.rng.random(len(data)) < prob).reshape(-1)
        noisy_output = np.zeros_like(data)

        noisy_output[inds], _ = self.PM_batch(data[inds])

        return noisy_output * self.d / self.k, inds
    
    def PM_batch(self, data):
        # print(f'pm: {data}')
        data = data.reshape(-1) / self.range
        noisy_output = np.zeros_like(data)

        u = self.rng.random(len(noisy_output))

        l = (np.exp(self.eps_k) + self.t) * (data * self.t - 1) / (self.t * (np.exp(self.eps_k) - 1))
        r = (np.exp(self.eps_k) + self.t) * (data * self.t + 1) / (self.t * (np.exp(self.eps_k) - 1))

        inner_idx = np.argwhere(u < np.exp(self.eps_k) / (self.t + np.exp(self.eps_k))).reshape(-1)
        outer_idx = np.argwhere(u >= np.exp(self.eps_k) / (self.t + np.exp(self.eps_k))).reshape(-1)

        inner_y = self.rng.random(len(inner_idx))
        # print(inner_y.shape)
        # print((r[inner_idx]-l[inner_idx]).shape)
        inner_y = (r[inner_idx]-l[inner_idx])*inner_y + l[inner_idx]
        # print(inner_y.shape)
        noisy_output[inner_idx] = inner_y

        length_l = np.abs(l[outer_idx] + self.A)
        legnth_r = np.abs(self.A - r[outer_idx])
        interval_l = length_l / (length_l + legnth_r)
        interval_random = self.rng.random(len(outer_idx))
        left_idx = outer_idx[interval_random < interval_l]
        right_idx = outer_idx[interval_random >= interval_l]
        
        left_y = self.rng.random(len(left_idx))
        left_y = (l[left_idx] + self.A) * left_y - self.A
        noisy_output[left_idx] = left_y

        right_y = self.rng.random(len(right_idx))
        right_y = (self.A - r[right_idx]) * right_y + r[right_idx]
        noisy_output[right_idx] = right_y

        return noisy_output * self.range, None

    def HM_single(self, x):
        if self.rng.random() > self.beta:
            y = self.TO_single(x)
        else:
            y = self.PM_single(x)

        return y
    
    # def HM_multi_batch(self, data):
    #     sampled_dims = np.random.choice(len(data), self.k, replace=False)
    #     noisy_output = np.zeros_like(data)

    #     noisy_output[sampled_dims], _ = self.HM_batch(data[sampled_dims])
    #     # no_inds = inds[no_pm_inds[0]]
    #     # pm_inds = inds[no_pm_inds[1]]
    #     return noisy_output * self.d / self.k
    
    def HM_batch(self, data):
        data = data.reshape(-1)
        hm_random = self.rng.random(len(data))
        to_inds = np.argwhere(hm_random > self.beta).reshape(-1)
        pm_inds = np.argwhere(hm_random <= self.beta).reshape(-1)
        noisy_output = np.zeros_like(data)

        noisy_output[to_inds], _ = self.TO_batch(data[to_inds])
        noisy_output[pm_inds], _ = self.PM_batch(data[pm_inds])

        return noisy_output, [to_inds, pm_inds]
    
    def HM_multi_batch(self, sampled_values, row_indices, sampled_col_indices, num_sam, d, k):
        perturbed_values, hm_inds = self.HM_batch(sampled_values.flatten())
        perturbed_values = perturbed_values.reshape(num_sam, k)

        # 4. 결과 배열을 생성하고, 교란된 값들을 원래 위치에 한 번에 삽입합니다.
        noisy_data = np.zeros((num_sam, d))
        noisy_data[row_indices, sampled_col_indices] = perturbed_values
        
        # 5. 통계적 보정을 위해 d/k 스케일링을 적용합니다.
        return noisy_data * (d / k)
    
    def HM_var(self, x):
        a = self.P0_0
        b = self.P0_0 * (1 - 1/np.exp(self.eps_k))

        return self.beta * ((self.t+1) * x**2 / (np.exp(self.eps_k)-1) + (self.t +np.exp(self.eps_k)) * (((self.t+1)**3) + np.exp(self.eps_k) - 1) / (3*self.t**2 * (np.exp(self.eps_k) -1)**2)) + (1 - self.beta) *( ( (1 - a) * np.exp(2*self.eps_k) * (np.exp(self.eps_k) + 1 )**2 / ((np.exp(self.eps_k) - 1)**2 * (np.exp(self.eps_k)-a)**2)) + b * np.abs(x) * np.exp(2 * self.eps_k) * (np.exp(self.eps_k) + 1 )**2 / ( (np.exp(self.eps_k) - 1)**2 * (np.exp(self.eps_k) - a) **2 ) - x**2 )
    
    # def HM_var(self, x):
    #     b = self.P0_0 * (1 - 1/np.exp(self.eps_k))
    #     C = np.exp(self.eps_k) * (np.exp(self.eps_k) +1) / ((np.exp(self.eps_k) -1) * np.exp(self.eps_k) - self.P0_0)
    #     return  self.beta * ((self.t+1) * x**2 / (np.exp(self.eps_k) - 1) + (self.t + np.exp(self.eps_k)) * ((self.t+ 1)**3 + np.exp(self.eps_k) -1) / (3 * self.t**2 * (np.exp(self.eps_k) -1))) + (1-self.beta) * ( (1-self.P0_0) * C**2 + C**2 * b * np.abs(x) - x**2 )
        
    
    def HM_max_var(self):
        # print(self.beta)

        b = self.beta
        v = self.HM_var(self.x_star)
        V0 = self.HM_var(0)
        V1 = self.HM_var(1)
        V = max(V0, V1)
        V = max(V, v)

        return V

    def HM_multi(self, x, m=None):
        original_shape = x.shape
        x = x.reshape(self.d)
        y = np.zeros_like(x)

        # if self.topk:
        #     js = np.argpartition(np.abs(x), -self.k)[-self.k:]

        # else:
        js = self.rng.choice([i for i in range(self.d)], self.k, False, p=[1/self.d for i in range(self.d)])
        
        for j in js:
            noisy_output = self.HM_single(x[j])
            # if m is not None and is_pm:
            #     noisy_output = utils.discretization(self.A, noisy_output / self.range, self.rng, m) * self.range
            y[j] = self.d * noisy_output / self.k

        return y.reshape(original_shape)
    
    def HM_multi_simul(self, x, m=None):
        prob = self.k / self.d
        if np.random.rand() < prob:
            y = self.HM_single(x)
        else:
            y = 0

        return y * self.d / self.k
    
    def HM_multi_simul_batch(self, data):
        data = data.reshape(-1)
        prob = self.k / self.d
        noisy_output = np.zeros_like(data)

        inds = np.argwhere(self.rng.random(len(data)) < prob).reshape(-1)
        noisy_output[inds], to_pm_inds = self.HM_batch(data[inds])
        to_inds = inds[to_pm_inds[0]]
        pm_inds = inds[to_pm_inds[1]]
        return noisy_output * self.d / self.k, [to_inds, pm_inds]
    
    def TO_worst_case_batch(self, num):
        worst_x = self.C**2 * self.P0_0 * (1 - 1/np.exp(self.eps)) / 2
        data = np.full(num, worst_x)
        perturbed, _ = self.TO_batch(data)
        perturbed_mean = np.mean(perturbed)

        return worst_x, perturbed_mean
    
    def get_worst_case(self):
        worst_x = {}
        worst_x['to'] = self.C**2 * self.P0_0 * (1 - 1/np.exp(self.eps_k)) / 2
        worst_x['pm_sub'] = 1
        worst_x['topm'] = self.x_star
        return worst_x
    
    def PM_worst_case_batch(self, num):
        worst_x = 1
        data = np.full(num, worst_x)
        perturbed, _ = self.PM_batch(data)
        perturbed_mean = np.mean(perturbed)

        return worst_x, perturbed_mean
    
    def HM_worst_case_batch(self, num):
        worst_x = self.x_star
        data = np.full(num, worst_x)
        perturbed, _ = self.HM_batch(data)
        perturbed_mean = np.mean(perturbed)

        return worst_x, perturbed_mean
    
    def exp_worst_case(self, name, num):
        if name == 'to':
            return self.TO_worst_case_batch(num)
        elif name == 'topm':
            return self.HM_worst_case_batch(num)
        elif name == 'pm_sub':
            return self.PM_worst_case_batch(num)
    
    def exp_batch(self, name, data):
        if name == 'to':
            return self.TO_batch(data)
        elif name =='topm':
            return self.HM_batch(data)
        elif name in ['pm_sub', 'pm_sub_16', 'pm_sub_4', 'pm_sub_8']:
            return self.PM_batch(data)
        
    def exp_multi_batch(self, name, sampled_values, row_indices, sampled_col_indices, num_sam, d, k):
        if name == 'to':
            return self.TO_multi_batch(sampled_values, row_indices, sampled_col_indices, num_sam, d, k)
        elif name == 'topm':
            return self.HM_multi_batch(sampled_values, row_indices, sampled_col_indices, num_sam, d, k)
        elif name in ['pm_sub', 'pm_sub_16', 'pm_sub_4', 'pm_sub_8']:
            return self.PM_multi_batch(sampled_values, row_indices, sampled_col_indices, num_sam, d, k)
        # elif name == 'topm':
        #     return self.NO_multi_batch(sampled_values, row_indices, sampled_col_indices, num_sam, d, k)
    
    def exp_multi(self, name, data):
        if name == 'to':
            return self.TO_multi_simul_batch(data)
        elif name == 'topm':
            return self.HM_multi_simul_batch(data)
        elif name in ['pm_sub', 'pm_sub_16', 'pm_sub_4', 'pm_sub_8']:
            return self.PM_multi_simul_batch(data)
        
    def multi(self, name, data):
        if name == 'to':
            return self.TO_multi_batch(data)
        elif name == 'topm':
            return self.HM_multi_batch(data)
        elif name in ['pm_sub', 'pm_sub_16', 'pm_sub_4', 'pm_sub_8']:
            return self.PM_multi_batch(data)
        

def vectorized_discretization(C, y_batch, m, rng):
    """
    요청하신 이산화 로직의 고속 벡터화 버전입니다.
    연속 노이즈 값(y_batch)을 받아 이산화된 값(z_batch)을 반환합니다.

    Args:
        C (float): 메커니즘의 최대 출력 범위 (e.g., mechanism.A)
        y_batch (np.array): (N,) shape의 연속적인 노이즈 값 배열 ([-C, C] 범위)
        m (int): 한쪽 방향의 이산화 구간 수 (e.g., 4 또는 8)
        rng (np.random.Generator): NumPy의 랜덤 생성기
        
    Returns:
        np.array: (N,) shape의 이산화된 노이즈 값 배열
    """
    # 0으로 나누기 방지
    C = max(C, 1e-9)
    m = int(m)
    y_batch = np.asarray(y_batch)

    # 1. y값의 범위를 [-m, m]으로 스케일링
    y_scaled = m * y_batch / C

    # 2. 왼쪽 이산점의 인덱스 k 계산 (e.g., y=0.1C, m=4 -> y_scaled=0.4, k=0)
    k = np.floor(y_scaled)

    # 3. 경계값 엣지 케이스 처리
    # k+1이 m을 초과하지 않도록 (y=C일 때 k=m이 될 수 있음)
    k = np.clip(k, -m, m - 1) 
    # y=C일 때, k=m-1로 설정 (오른쪽 점 m 선택)
    k[y_batch == C] = m - 1 
    # y=-C일 때, k=-m로 설정 (왼쪽 점 -m 선택)
    k[y_batch == -C] = -m 

    # 4. 오른쪽 점(z_k+1)을 선택할 확률 P_right 계산
    # P_right = (y - z_k) / (z_k+1 - z_k) = (y - Ck/m) / (C/m) = my/C - k
    prob_right = y_scaled - k
    # y=-C일 때, k=-m이므로 prob_right=0이 되어 항상 왼쪽 점(-m) 선택
    prob_right[y_batch == -C] = 0 

    # 5. 랜덤 샘플링
    u = rng.random(size=y_batch.shape)

    # 6. u < P_right 이면 오른쪽 점(k+1)을, 아니면 왼쪽 점(k)을 선택
    chosen_k = k + (u < prob_right).astype(int)

    # 7. 최종 이산값 z 계산
    # z = C * k_chosen / m
    return C * chosen_k / m

class DiscretizedMechanismWrapper:
    """
    PM 또는 PM-sub 메커니즘을 감싸(wrap)서 출력을 이산화합니다.
    평균 추정(Mean Estimation) 전용입니다.
    base_mechanism 객체의 PM_batch 메서드를 오버라이드합니다.
    """
    def __init__(self, base_mechanism, m):
        self.base_mech = base_mechanism
        self.m = int(m)
        self.A = base_mechanism.A 
        self.rng = base_mechanism.rng

        print('crete discrete') # <-- 이것이 먼저 출력되어야 합니다.
        
        import functools
        for attr in dir(self.base_mech):
            if not attr.startswith('__') and not hasattr(self, attr):
                # [수정] 오버라이드 할 모든 메서드를 복사 대상에서 제외합니다.
                if attr in ['PM_batch', 'PM_multi_batch', 'exp_multi_batch']: 
                    continue
                setattr(self, attr, getattr(self.base_mech, attr))

    # [신규] exp_multi_batch를 래퍼가 직접 구현 (오버라이드)
    def exp_multi_batch(self, name, sampled_values, row_indices, sampled_col_indices, num_sam, d, k):
        """
        [오버라이드] 실험 스크립트의 진입점입니다.
        pm_sub 계열의 호출을 래퍼의 PM_multi_batch로 보냅니다.
        """
        # print(f"!!! Wrapper.exp_multi_batch CALLED with name={name}") # (디버그용)
        
        if name in ['pm_sub', 'pm_sub_16', 'pm_sub_4', 'pm_sub_8']:
            # [중요] base_mech가 아닌 self (래퍼)의 PM_multi_batch 호출
            return self.PM_multi_batch(sampled_values, row_indices, sampled_col_indices, num_sam, d, k)
        else:
            # pm_sub가 아닌 다른 호출(e.g., 'to', 'topm')은 원본 객체로 전달
            return self.base_mech.exp_multi_batch(name, sampled_values, row_indices, sampled_col_indices, num_sam, d, k)

    # [신규] PM_multi_batch를 래퍼가 직접 구현 (오버라이드)
    def PM_multi_batch(self, sampled_values, row_indices, sampled_col_indices, num_sam, d, k):
        """
        [오버라이드] 이 메서드가 래퍼의 PM_batch를 호출하도록 보장합니다.
        """
        # print("!!! Wrapper.PM_multi_batch CALLED") # (디버그용)
        
        # [중요] base_mech가 아닌 self (래퍼)의 PM_batch 호출
        perturbed_values, _ = self.PM_batch(sampled_values.flatten())
        perturbed_values = perturbed_values.reshape(num_sam, k)

        # 4. 결과 배열을 생성하고, 교란된 값들을 원래 위치에 한 번에 삽입합니다.
        noisy_data = np.zeros((num_sam, d))
        noisy_data[row_indices, sampled_col_indices] = perturbed_values
        
        # 5. 통계적 보정을 위해 d/k 스케일링을 적용합니다.
        return noisy_data * (d / k)

    def PM_batch(self, data):
        """
        [오버라이드] 기본 메커니즘의 PM_batch를 호출하고, 그 결과를 이산화합니다.
        """
        # 1. 기본 메커니즘에서 연속 노이즈 y 생성
        # (self.base_mech는 pm.PM 또는 topm.TOPM 인스턴스)
        y_continuous, _ = self.base_mech.PM_batch(data) # --> "pm batch" 출력
        
        print('discrete') # <-- [드디어 출력될] 이산화 시작점
        
        # 2. y를 다시 이산화 z로 변환
        z_discrete = vectorized_discretization(
            self.A, 
            y_continuous, 
            self.m, 
            self.rng
        )
        print(z_discrete) # (이산화된 값 출력)
        return z_discrete, None


if __name__ == '__main__':
    eps = 5
    # print(eps)
    rng = np.random.default_rng(seed=22)
    no = TOPM(1, eps, rng)

    sum = 0
    for i in range(100):
        data = np.random.uniform(-1, 1, (100000,))
        mean = np.mean(data)
        noisy, _ = no.TO_batch(data)
        nm = np.mean(noisy)
        # print(f'men: {mean}, mn: {nm}')
        sum += (nm - mean)**2
    print(f'err: {sum / 100}')


    eps = 5
    # print(eps)
    rng = np.random.default_rng(seed=22)
    no = TOPM(1, eps, rng)

    sum = 0
    for i in range(100):
        data = np.random.uniform(-1, 1, (100000,))
        mean = np.mean(data)
        noisy, _ = no.PM_batch(data)
        nm = np.mean(noisy)
        # print(f'men: {mean}, mn: {nm}')
        sum += (nm - mean)**2
    print(f'err: {sum / 100}')


    # for eps in [5.0]:
    #     rng = np.random.default_rng(seed=0)
    #     topm = TOPM(1, eps, rng)
    #     print(f'eps: {eps}, var: {topm.TO_max_var()}')
    #     print(f'p0: {topm.P0_0 / np.exp(eps)}, C: {topm.C}')
    #     print(f'worstcase x: {topm.C**2 * topm.P0_0 * (1 - 1/np.exp(topm.eps)) / 2}')
    #     # print(f'eps: {eps}, var: {topm.HM_max_var()}')
    # # print(topm.HM_max_var())
