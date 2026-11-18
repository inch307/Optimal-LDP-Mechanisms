import numpy as np
from scipy.optimize import minimize, Bounds
import time
import argparse
import pickle
import os

# ===================================================================
# 헬퍼 함수: Worst-Case 분산 계산 (변경 없음)
# ===================================================================
def calculate_worst_case_variance(v, optimizer):
    n, a, p = optimizer.n, *optimizer.unpack_vars(v)
    a_sq = a**2
    x, E_Y2 = a @ p, a_sq @ p
    variances_at_xj = E_Y2 - x**2
    candidate_variances = list(variances_at_xj)
    for j in range(1, n + 1):
        xj, xjm1 = x[j], x[j - 1]; dx = xj - xjm1
        if abs(dx) < 1e-12: continue
        E_Y2_j, E_Y2_jm1 = E_Y2[j], E_Y2[j-1]
        x_star_num, x_star_den = E_Y2_j - E_Y2_jm1, 2 * dx
        if abs(x_star_den) < 1e-12: continue
        x_star = x_star_num / x_star_den
        if (x_star >= xjm1 - 1e-9) and (x_star <= xj + 1e-9):
            slope_E_Y2 = x_star_num / dx
            E_Y2_at_x_star = slope_E_Y2 * (x_star - xj) + E_Y2_j
            var_at_x_star = E_Y2_at_x_star - x_star**2
            candidate_variances.append(var_at_x_star)
    return np.max(candidate_variances)

# ===================================================================
# ✨✨✨ 버전 7: a_1, ..., a_n 모두 최적화, a_0 포함 (N이 홀수일 때)
# ===================================================================
class WorstCaseOptimizerWithA0_FreeA:
    def __init__(self, n, epsilon):
        self.n, self.epsilon, self.exp_eps = n, epsilon, np.exp(epsilon)
        # === [수정] === a 변수의 개수가 1개에서 n개로 변경
        self.num_a_vars, self.num_p_vars = n, (2 * n + 1) * (n + 1)
        print(f"Worst-Case 옵티마이저 생성됨 (모든 a_i 최적화, a_0 포함, n={n})")

    def unpack_vars(self, v):
        # === [수정] === v의 첫 n개 요소를 a_pos로 직접 사용
        a_pos = v[:self.num_a_vars]
        p_v = v[self.num_a_vars:].reshape((self.n + 1, 2 * self.n + 1))
        a_full = np.concatenate([-a_pos[::-1], [0], a_pos])
        return a_full, p_v.T
    
    def get_p_val(self, p, i, j):
        if j < 0: return p[self.n - i, -j]
        return p[self.n + i, j]

    def objective_func(self, v):
        return calculate_worst_case_variance(v, self)

    def _create_constraints(self):
        n = self.n; cons = []
        MIN_GAP = 1e-7 # a와 x 값들의 최소 간격

        cons.append({'type': 'eq', 'fun': lambda v: np.sum(v[self.num_a_vars:].reshape((n + 1, 2 * n + 1)), axis=1) - 1})
        cons.append({'type': 'eq', 'fun': lambda v: self.unpack_vars(v)[0] @ self.unpack_vars(v)[1][:, n] - 1})
        
        # === [수정] === a_i들의 제약조건
        # a_1 > 0
        if n > 0: cons.append({'type': 'ineq', 'fun': lambda v: v[0] - MIN_GAP})
        # a_i < a_{i+1}
        if n > 1: cons.append({'type': 'ineq', 'fun': lambda v: np.diff(v[:n]) - MIN_GAP})
        
        cons.append({'type': 'ineq', 'fun': lambda v: np.diff(self.unpack_vars(v)[0] @ self.unpack_vars(v)[1]) - MIN_GAP})
        
        def p_symmetry_j0(v):
            _, p = self.unpack_vars(v)
            return np.array([self.get_p_val(p, i, 0) - self.get_p_val(p, -i, 0) for i in range(1, n + 1)])
        cons.append({'type': 'eq', 'fun': p_symmetry_j0})
        
        def ldp_constraints(v):
            _, p = self.unpack_vars(v); constraints = []
            for i in range(1, n + 1):
                p_neg_i_n = self.get_p_val(p, -i, n)
                for j in range(-n, n + 1):
                    p_i_j = self.get_p_val(p, i, j)
                    constraints.extend([p_i_j - p_neg_i_n, self.exp_eps * p_neg_i_n - p_i_j])
            p_zero_n = self.get_p_val(p, 0, n)
            for j in range(-n, n + 1):
                p_zero_j = self.get_p_val(p, 0, j)
                constraints.extend([p_zero_j - p_zero_n, self.exp_eps * p_zero_n - p_zero_j])
            return np.array(constraints)
        cons.append({'type': 'ineq', 'fun': ldp_constraints})
        return cons

    def run(self, initial_guess, options):
        # Bounds 설정은 num_a_vars가 변경됨에 따라 자동으로 적용됨
        lv = [0.0] * self.num_a_vars + [0.0] * self.num_p_vars
        uv = [np.inf] * self.num_a_vars + [1.0] * self.num_p_vars
        b, c = Bounds(lv, uv), self._create_constraints()
        cb = SimpleCallbackLogger(self)
        print(f"\nn={self.n},eps={self.epsilon} Worst-Case 최적화 시작 (모든 a_i, a_0 포함)...")
        return minimize(self.objective_func, initial_guess, method='SLSQP', bounds=b, constraints=c, options=options, callback=cb)

# ===================================================================
# ✨✨✨ 버전 8: a_1, ..., a_n 모두 최적화, a_0 제외 (N이 짝수일 때)
# ===================================================================
class WorstCaseOptimizerNoA0_FreeA:
    def __init__(self, n, epsilon):
        self.n, self.epsilon, self.exp_eps = n, epsilon, np.exp(epsilon)
        self.num_output_points = 2 * n
        # === [수정] === a 변수의 개수가 1개에서 n개로 변경
        self.num_a_vars, self.num_p_vars = n, self.num_output_points * (n + 1)
        print(f"Worst-Case 옵티마이저 생성됨 (모든 a_i 최적화, a_0 제외, n={n})")

    def _map_i_to_idx(self, i):
        return i + self.n if i < 0 else i + self.n - 1

    def unpack_vars(self, v):
        # === [수정] === v의 첫 n개 요소를 a_pos로 직접 사용
        a_pos = v[:self.num_a_vars]
        p_v = v[self.num_a_vars:].reshape((self.n + 1, self.num_output_points))
        a_full = np.concatenate([-a_pos[::-1], a_pos])
        return a_full, p_v.T

    def get_p_val(self, p, i, j):
        if j < 0: return p[self._map_i_to_idx(-i), -j]
        return p[self._map_i_to_idx(i), j]

    def objective_func(self, v):
        return calculate_worst_case_variance(v, self)

    def _create_constraints(self):
        n = self.n; cons = []
        MIN_GAP = 1e-7

        cons.append({'type': 'eq', 'fun': lambda v: np.sum(v[self.num_a_vars:].reshape((n + 1, 2 * n)), axis=1) - 1})
        cons.append({'type': 'eq', 'fun': lambda v: self.unpack_vars(v)[0] @ self.unpack_vars(v)[1][:, n] - 1})
        
        # === [수정] === a_i들의 제약조건
        if n > 0: cons.append({'type': 'ineq', 'fun': lambda v: v[0] - MIN_GAP})
        if n > 1: cons.append({'type': 'ineq', 'fun': lambda v: np.diff(v[:n]) - MIN_GAP})
        
        cons.append({'type': 'ineq', 'fun': lambda v: np.diff(self.unpack_vars(v)[0] @ self.unpack_vars(v)[1]) - MIN_GAP})
        
        def p_symmetry_j0(v):
            _, p = self.unpack_vars(v)
            return np.array([self.get_p_val(p, i, 0) - self.get_p_val(p, -i, 0) for i in range(1, n + 1)])
        cons.append({'type': 'eq', 'fun': p_symmetry_j0})

        def ldp_constraints(v):
            _, p = self.unpack_vars(v); constraints = []
            for i in range(1, n + 1):
                p_neg_i_n = self.get_p_val(p, -i, n)
                for j in range(-n, n + 1):
                    p_i_j = self.get_p_val(p, i, j)
                    constraints.extend([p_i_j - p_neg_i_n, self.exp_eps * p_neg_i_n - p_i_j])
            return np.array(constraints)
        cons.append({'type': 'ineq', 'fun': ldp_constraints})
        return cons

    def run(self, initial_guess, options):
        num_a, num_p = self.num_a_vars, self.num_p_vars
        lower, upper = [0.0] * num_a + [0.0] * num_p, [np.inf] * num_a + [1.0] * num_p
        b, c = Bounds(lower, upper), self._create_constraints()
        cb = SimpleCallbackLogger(self)
        print(f"\nn={self.n},eps={self.epsilon} Worst-Case 최적화 시작 (모든 a_i, a_0 제외)...")
        return minimize(self.objective_func, initial_guess, method='SLSQP', bounds=b, constraints=c, options=options, callback=cb)

# ===================================================================
# ✨✨✨ [수정] 초기값 생성 함수가 a_1, ..., a_n 배열을 생성하도록 변경
# ===================================================================
def create_initial_guess_free_a(n, epsilon, use_a0, smoothing_factor=0.1):
    print(f"n={n}, epsilon={epsilon} PM 초기값 계산 (모든 a_i 최적화 버전)...")
    if n == 0: raise ValueError("n=0은 지원하지 않습니다.")
    
    # 먼저 기준이 되는 a_1 값을 계산
    exp_eps = np.exp(epsilon)
    initial_a_1_base = (exp_eps + 1) / ((exp_eps - 1) * n)
    
    # i * a_1 방식으로 초기 a_pos 배열 생성
    initial_a_pos = initial_a_1_base * np.arange(1, n + 1)
    
    # p_matrix 생성 로직은 a_1 기준 값(A)을 사용하므로 거의 동일
    A = n * initial_a_1_base
    t = np.exp(epsilon/3); p_core, p_tail = exp_eps/(exp_eps+t), t/(exp_eps+t)
    get_overlap = lambda a, b: max(0, min(a[1], b[1]) - max(a[0], b[0]))
    num_output = 2 * n + 1 if use_a0 else 2 * n
    p_matrix = np.zeros((n + 1, num_output))
    i_range = range(-n, n + 1) if use_a0 else list(range(-n,0))+list(range(1,n+1))
    
    for j in range(n + 1):
        x = j / n; l_x, r_x = (A+1)/2.*x-(A-1)/2., (A+1)/2.*x+(A-1)/2.
        w_c, w_t = r_x - l_x, 2*A - (r_x - l_x)
        pdf_c = p_core/w_c if w_c > 1e-9 else 0; pdf_t = p_tail/w_t if w_t > 1e-9 else 0
        for idx, i in enumerate(i_range):
            if i == -n: s, e = -A, (-n + 0.5) / n * A
            elif i == n: s, e = (n - 0.5) / n * A, A
            elif i == 0 and use_a0: s, e = (-0.5 / n * A, 0.5 / n * A)
            else: s, e = (i - 0.5) / n * A, (i + 0.5) / n * A
            p_matrix[j, idx] = get_overlap([s,e],[-A,l_x])*pdf_t+get_overlap([s,e],[l_x,r_x])*pdf_c+get_overlap([s,e],[r_x,A])*pdf_t
        row_sum = np.sum(p_matrix[j, :]);
        if row_sum > 1e-9: p_matrix[j, :] /= row_sum
        
    # x_n=1 제약 만족을 위한 스케일링
    a_full_initial = np.concatenate([-initial_a_pos[::-1], [0], initial_a_pos]) if use_a0 else np.concatenate([-initial_a_pos[::-1], initial_a_pos])
    x_n_initial = np.dot(p_matrix[n, :], a_full_initial)
    if abs(x_n_initial) > 1e-9:
        scaling_factor = 1.0 / x_n_initial
        initial_a_pos *= scaling_factor # a_pos 배열 전체를 스케일링
        print(f"  [INFO] 초기 a_i 값들을 {scaling_factor:.4f}배 스케일링하여 x_n=1 제약 만족.")
        
    uniform = np.full_like(p_matrix, 1./num_output)
    smoothed = (1-smoothing_factor)*p_matrix + smoothing_factor*uniform
    for j in range(n+1):
        row_sum = np.sum(smoothed[j,:]);
        if row_sum > 1e-9: smoothed[j,:] /= row_sum
        
    # 최종 초기 추정치 벡터: [a_1, ..., a_n, p_1,1, p_1,2, ...]
    return np.concatenate([initial_a_pos, smoothed.flatten()])

def create_initial_guess_pm_noise(n, epsilon, use_a0, noise_scale=0.05):
    """
    기존 PM 스타일 초기값에 약간의 노이즈를 추가합니다.
    - a: 값에 변동을 준 후, np.sort()를 통해 오름차순(a_i < a_{i+1}) 제약 유지
    - p: 값에 변동을 준 후, 행별 정규화를 통해 sum=1 제약 유지
    """
    print(f"  [INFO] 'pm_noise' 초기값 생성 (noise_scale={noise_scale})...")
    
    # 1. 깨끗한 PM 초기값 생성
    # (내부적으로 create_initial_guess_free_a 사용)
    clean_v = create_initial_guess_free_a(n, epsilon, use_a0)
    
    # 2. 변수 분리
    a_clean = clean_v[:n]
    p_clean = clean_v[n:]
    
    # 3. a에 노이즈 추가 (Multiplicative Noise)
    # a_new = a_old * (1 + random_noise)
    # 0.95 ~ 1.05 배 수준의 변동
    noise_a = np.random.uniform(-noise_scale, noise_scale, size=n)
    a_noisy = a_clean * (1.0 + noise_a)
    
    # [중요] 제약조건 강제: 양수 & 오름차순 정렬
    a_noisy = np.abs(a_noisy) # 혹시 모를 음수 방지
    a_noisy.sort()            # a_i < a_{i+1} 강제
    
    # 4. p에 노이즈 추가 (Additive Noise)
    # 확률 매트릭스로 변환
    num_outputs = len(p_clean) // (n + 1)
    p_mat = p_clean.reshape((n + 1, num_outputs))
    
    # 약간의 양수 노이즈를 더함 (음수 확률 방지)
    noise_p = np.random.uniform(0, noise_scale * 0.1, size=p_mat.shape)
    p_noisy = p_mat + noise_p
    
    # [중요] 제약조건 강제: 각 행의 합 = 1
    row_sums = p_noisy.sum(axis=1, keepdims=True)
    p_noisy = p_noisy / row_sums
    
    return np.concatenate([a_noisy, p_noisy.flatten()])

def create_initial_guess_random_uniform(n, num_outputs):
    """
    무작위 'uniform' 초기값을 생성합니다.
    - a_i 값들은 (0, 5) 범위에서 무작위로 샘플링 후 정렬하여 a_i < a_{i+1} 제약 충족
    - p_matrix 값들은 (0, 1) 범위에서 무작위로 샘플링 후 각 행을 정규화하여 sum(p_j) = 1 제약 충족
    - (x_n=1 등 다른 제약은 의도적으로 만족시키지 않아, 옵티마이저의 견고성을 테스트)
    """
    print(f"  [INFO] 'random_uniform' 초기값 생성 (n={n}, num_outputs={num_outputs})")
    
    # 1. a_pos 생성 (n개)
    # 0.01 (최소 갭) ~ 5.0 사이에서 랜덤하게 생성 후 정렬
    # (a_i < a_{i+1} 제약 만족을 위해)
    a_pos_increments = np.random.uniform(0.01, 1.0, size=n)
    a_pos = np.cumsum(a_pos_increments)
    
    # 2. p_matrix 생성 ((n+1) x num_outputs 개)
    p_shape = (n + 1, num_outputs)
    p_matrix = np.random.rand(*p_shape)
    # 각 행의 합이 1이 되도록 정규화
    p_matrix = p_matrix / np.sum(p_matrix, axis=1, keepdims=True)
    
    # 3. 벡터로 결합
    return np.concatenate([a_pos, p_matrix.flatten()])

def create_initial_guess_uniform(n, num_outputs):
    """
    무작위성이 없는 'uniform' 초기값을 생성합니다.
    - a_i 값들은 1.0, 2.0, ..., n.0 으로 등간격 설정 (a_i < a_{i+1} 만족)
    - p_matrix 값들은 모두 1/num_outputs 로 균등 설정 (sum(p)=1 만족)
    """
    print(f"  [INFO] 'uniform' (deterministic) 초기값 생성 (n={n}, num_outputs={num_outputs})")
    
    # 1. a_pos 생성: [1.0, 2.0, ..., n.0]
    a_pos = np.arange(1, n + 1, dtype=float)
    
    # 2. p_matrix 생성: 모든 확률이 동일
    p_matrix = np.full((n + 1, num_outputs), 1.0 / num_outputs)
    
    # 3. 벡터로 결합
    return np.concatenate([a_pos, p_matrix.flatten()])

# (나머지 헬퍼 함수들은 변경 없음)
class SimpleCallbackLogger:
    def __init__(self, optimizer_instance, disp_interval=10):
        self.iteration = 0; self.optimizer = optimizer_instance
        self.last_fun_val = np.inf; self.disp_interval = disp_interval
        print("\n--- Optimization Log ---"); print("{:>5} | {:>18} | {:>15}".format("Iter", "Objective Value", "Change")); print("-" * 45)
    def __call__(self, xk):
        if self.iteration == 0 or (self.iteration + 1) % self.disp_interval == 0:
            fun_val = self.optimizer.objective_func(xk)
            change = self.last_fun_val - fun_val; self.last_fun_val = fun_val
            print("{:5d} | {:18.6f} | {:15.6e}".format(self.iteration, fun_val, change))
        self.iteration += 1

def calculate_average_case_variance(v, optimizer):
    n, a, p = optimizer.n, *optimizer.unpack_vars(v)
    x = a @ p; total = 0.0; a_sq = a**2
    for j in range(1, n + 1):
        xj, xjm1 = x[j], x[j - 1]; dx = xj - xjm1
        if abs(dx) < 1e-20: continue
        pj, pjm1 = p[:, j], p[:, j - 1]
        avg_variance_proxy = (np.dot(a_sq, pj) + np.dot(a_sq, pjm1)) / 2.0
        integral_of_P = avg_variance_proxy * dx
        integral_of_x_sq = (xj**3 - xjm1**3) / 3.0
        total += (integral_of_P - integral_of_x_sq)
    return total

def display_and_save_results(result, optimizer, N, output_dir='results_worst_case_free_a'):
    print("\n" + "#"*20 + f" n = {optimizer.n} 최종 최적화 결과 " + "#"*20)
    final_a, final_p = optimizer.unpack_vars(result.x)
    final_x = final_a @ final_p
    avg_case_val = calculate_average_case_variance(result.x, optimizer)

    print(f"\n--- 최종 'a' 값 (독립적으로 최적화됨) ---")
    print(f"a = {np.array2string(final_a, precision=6, max_line_width=120)}")

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    filename = os.path.join(output_dir, f"opt_results_worst_case_eps{optimizer.epsilon}_N{N}.pkl")
    
    results_data = {
        'metadata': { 'N_total_a_points': N, 'n_param': optimizer.n, 'epsilon': optimizer.epsilon,
                      'use_a0': isinstance(optimizer, WorstCaseOptimizerWithA0_FreeA),
                      'optimization_type': 'worst_case_variance_free_a', 'success': result.success, 
                      'message': result.message, 'worst_var': result.fun,
                      'avg_var': avg_case_val,
                      'iterations': result.nit },
        'a_values': final_a, 'p_matrix': final_p,
        'x_values': final_x, 'scipy_result_object': result }
    
    with open(filename, 'wb') as f: pickle.dump(results_data, f)
    print(f"\n결과가 '{filename}' 파일로 저장되었습니다.")
    print(f"  Success: {result.success}\n  Message: {result.message}");
    print(f"  Final Objective Value (Worst-Case Var): {result.fun:.6f}")
    print(f"  Corresponding Average-Case Variance: {avg_case_val:.6f}")


def optimize(epsilon, N, ftol=1e-6, maxiter=3000, disp=False, initial_guess_strategy='pm_style'):
    """
    [수정됨] N, eps, ftol 및 초기화 전략에 따라 최적화를 1회 실행하고 결과 객체를 반환합니다.
    """
    my_options = {'ftol': ftol, 'maxiter': maxiter, 'disp': disp}
    
    if N <= 1: raise ValueError("N must be > 1 for this script.")
    
    # 새로운 클래스와 초기값 생성 함수를 사용하도록 변경
    if N % 2 == 0:
        n_param = N // 2
        num_outputs = 2 * n_param
        use_a0 = False
        optimizer = WorstCaseOptimizerNoA0_FreeA(n_param, epsilon)
    else:
        n_param = (N - 1) // 2
        num_outputs = 2 * n_param + 1
        use_a0 = True
        optimizer = WorstCaseOptimizerWithA0_FreeA(n_param, epsilon)
        
    # --- [수정된] 초기값 생성 로직 ---
    if initial_guess_strategy == 'pm_style':
        initial_guess = create_initial_guess_free_a(n_param, epsilon, use_a0=use_a0)
    elif initial_guess_strategy == 'random_uniform':
        initial_guess = create_initial_guess_random_uniform(n_param, num_outputs)
    elif initial_guess_strategy == 'uniform': 
        initial_guess = create_initial_guess_uniform(n_param, num_outputs)
    elif initial_guess_strategy == 'pm_noise': 
        initial_guess = create_initial_guess_pm_noise(n_param, epsilon, use_a0=use_a0, noise_scale=0.1)
    else:
        raise ValueError(f"Unknown initial_guess_strategy: {initial_guess_strategy}")
    # --- [수정 완료] ---

    result = optimizer.run(initial_guess, options=my_options)
    
    # --- [수정] 결과를 저장하지 않고 객체 자체를 반환 ---
    # (기존 if result.success: ... else: ... 로직 모두 주석 처리 또는 삭제)
    
    return result

# def optimize(epsilon, N, ftol=1e-6, maxiter=3000, disp=False):
#     my_options = {'ftol': ftol, 'maxiter': maxiter, 'disp': disp}
    
#     if N <= 1: raise ValueError("N must be > 1 for this script.")
    
#     # 새로운 클래스와 초기값 생성 함수를 사용하도록 변경
#     if N % 2 == 0:
#         n_param = N // 2
#         optimizer = WorstCaseOptimizerNoA0_FreeA(n_param, epsilon)
#         initial_guess = create_initial_guess_free_a(n_param, epsilon, use_a0=False)
#     else:
#         n_param = (N - 1) // 2
#         optimizer = WorstCaseOptimizerWithA0_FreeA(n_param, epsilon)
#         initial_guess = create_initial_guess_free_a(n_param, epsilon, use_a0=True)

#     result = optimizer.run(initial_guess, options=my_options)

#     if result.success:
#         output_dir = 'results_worst_case_free_a'
#         display_and_save_results(result, optimizer, N, output_dir)
#         print("\n모든 과정이 성공적으로 완료되었습니다.")
#     else:
#         output_dir = 'results_worst_case_free_a_failed'
#         print("\n최적화 실패.")
#         display_and_save_results(result, optimizer, N, output_dir)

#     # if result.success:
#     #     output_dir = 'results_worst_case'
#     #     display_and_save_results(result, optimizer, N, output_dir)
#     #     print("\n모든 과정이 성공적으로 완료되었습니다.")
#     # else:
#     #     output_dir = 'results_worst_case_failed'
#     #     print("\n최적화 실패.")
#     #     display_and_save_results(result, optimizer, N, output_dir)
    
