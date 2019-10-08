import numpy as np

print("Going to use Friston2003 BOLD model...")

t_min = 20  # (s)
dt = 0.001  # (s)
n_min = int(np.round(t_min / dt))


def computeRequiredVectorLength(T):
    global t_min, dt, n_min
    n_min = int(np.round(t_min / dt))
    n_t = int(T/dt)
    return n_t - n_min


def BOLDModel(T, r):
    global t_min, dt, n_min
    # The Hemodynamic model with one simplified neural activity
    #     by Friston et al. 2003, Friston et al. 2000
    #
    # T          : total time (s)
    # resolution : dt in (s)
    #
    # Code from Deco et al. 2014

    global itaus, itauf, itauo, ialpha

    # BOLD model parameters
    taus = 0.65    # 0.8;    % time unit (s)
    tauf = 0.41    # 0.4;    % time unit (s)
    tauo = 0.98    # 1;      % mean transit time (s)
    alpha = 0.33    #0.32; % 0.2;    % a stiffness exponent
    itaus = 1 / taus
    itauf = 1 / tauf
    itauo = 1 / tauo
    ialpha = 1 / alpha
    Eo = 0.34    # 0.8;    % resting oxygen extraction fraction  --> rho in the paper
    vo = 0.02    # --> V0, from Friston et al. 2003
    k1 = 7 * Eo # coeffs from Deco et al 2013
    k2 = 2
    k3 = 2 * Eo - 0.2

    # dt = resolution    # (s)
    #t0 = np.arange(0,T,dt).reshape(-1,1)
    #n_t = t0.size
    n_t = int(T/dt)

    # t_min = tmin
    n_min = int(np.round(t_min / dt))

    # Initial conditions
    x0 = np.array([0, 1, 1, 1])

    # Euler method
    t = t0
    x = np.zeros([n_t, 4])
    x[0,:] = x0
    for n in range(n_t-1):
        x[n + 1, 0] = x[n, 0] + dt * (r[n] - itaus * x[n, 0] - itauf * (x[n, 1] - 1))
        x[n + 1, 1] = x[n, 1] + dt * x[n, 0]
        x[n + 1, 2] = x[n, 2] + dt * itauo * (x[n, 1] - x[n, 2] ** ialpha)
        x[n + 1, 3] = x[n, 3] + dt * itauo * (x[n, 1] * (1 - (1 - Eo) ** (1 / x[n, 1])) / Eo - (x[n, 2] ** ialpha) * x[n, 3] / x[n, 2])

    # The Balloon-Windkessel model from Buxton et al. 1998:
    t = t[n_min:t.size]
    s = x[n_min:n_t, 0]
    fi = x[n_min:n_t, 1]
    v = x[n_min:n_t, 2]
    q = x[n_min:n_t, 3]
    b = 100 / Eo * vo * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))
    #clear(x)

    time = (np.arange(1,b.size+1)) * resolution + t_min * resolution

    return b  #, time

