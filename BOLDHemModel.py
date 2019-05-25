import numpy as np


def Model_Friston2003(T, r,
                      resolution=0.001, # (s)
                      tmin=None):

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
    Eo = 0.34    # 0.8;    % resting oxygen extraction fraction
    vo = 0.02
    k1 = 7 * Eo # coeffs from Deco et al 2013
    k2 = 2
    k3 = 2 * Eo - 0.2

    dt = resolution    # (s)
    t0 = np.arange(0,T+dt,dt).reshape(-1,1)
    n_t = t0.size

    if not tmin:
        t_min = dt
        n_min = 0
    else:
        t_min = tmin
        n_min = np.round(t_min / dt)

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

    return b #, time

def Model_Stephan2007(T, r,
                      resolution=0.001, # (s)
                      tmin=None):

    # The Hemodynamic model with one simplified neural activity
    #     by stephan et al. 2007
    #
    # T          : total time (s)
    # resolution : dt in (s)
    #
    # Code from Deco et al. 2018

    global itaus, itauf, itauo, ialpha, Eo, dt

    # BOLD model parameters
    taus = 0.65  # 0.8;    % time unit (s)
    tauf = 0.41  # 0.4;    % time unit (s)
    tauo = 0.98  # 1;      % mean transit time (s)
    alpha = 0.32 #0.32; % 0.2;    % a stiffness exponent
    itaus = 1 / taus
    itauf = 1 / tauf
    itauo = 1 / tauo
    ialpha = 1 / alpha
    Eo=0.4
    TE=0.04
    vo=0.04
    k1=4.3*40.3*Eo*TE
    k2=25*Eo*TE
    k3=1

    dt = resolution    # (s)
    t0 = np.arange(0,T+dt,dt).reshape(-1,1)
    n_t = t0.size

    if not tmin:
        t_min = dt
        n_min = 0
    else:
        t_min = tmin
        n_min = np.round(t_min / dt)

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
    b = vo * (k1 * (1 - q) + k2 * (1 - q / v) + k3 * (1 - v))

    return b
