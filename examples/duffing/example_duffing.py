import numpy as np
from scipy.integrate import odeint
import sys
sys.path.append('../../src')
from sindy_utils import library_size
from scipy.special import legendre


def get_duffing_data(n_ics, noise_strength=0):
    """
    Generate a set of duffing training data for multiple random initial conditions.

    Arguments:
        n_ics - Integer specifying the number of initial conditions to use.
        noise_strength - Amount of noise to add to the data.

    Return:
        data - Dictionary containing elements of the dataset. See generate_duffing_data()
        doc string for list of contents.
    """
    d = 2
    t = np.arange(0, 5, .02)
    n_steps = t.size
    input_dim = 128

    ic_means = np.array([0,25])
    ic_widths = 2*np.array([36,48])

    # training data
    ics = ic_widths*(np.random.rand(n_ics, d)-.5) + ic_means
    data = generate_duffing_data(ics, t, input_dim, linear=False, normalization=np.array([1/40,1/40]))
    data['x'] = data['x'].reshape((-1,input_dim)) + noise_strength*np.random.randn(n_steps*n_ics,input_dim)
    data['dx'] = data['dx'].reshape((-1,input_dim)) + noise_strength*np.random.randn(n_steps*n_ics,input_dim)
    data['ddx'] = data['ddx'].reshape((-1,input_dim)) + noise_strength*np.random.randn(n_steps*n_ics,input_dim)

    return data


def simulate_duffing(z0, t, alpha=1., beta=5., gamma=8., delta=10., omega=0.5):
    """
    Simulate the duffing dynamics.

    Arguments:
        z0 - Initial condition in the form of a 2-value list or array.
        t - Array of time points at which to simulate.
        alpha, beta, gamma, delta, omega - duffing parameters

    Returns:
        z, dz, ddz - Arrays of the trajectory values and their 1st and 2nd derivatives.
    """
    f = lambda z,t : [z[1], -delta*z[1] - alpha*z[0] - beta*z[0]**3 + gamma*np.cos(omega*t)]
    df = lambda z,dz,t : [dz[1],
                          -delta*dz[1] - alpha*dz[0] - 3*beta*z[0]**2*dz[0] + gamma*omega*np.sin(omega*t)]

    z = odeint(f, z0, t)

    dt = t[1] - t[0]
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(t.size):
        dz[i] = f(z[i],dt*i)
        ddz[i] = df(z[i], dz[i], dt*i)
    return z, dz, ddz


def generate_duffing_data(ics, t, n_points, linear=True, normalization=None,
                            alpha=1., beta=5., gamma=8., delta=10., omega=0.5):
    """
    Generate high-dimensional duffing data set.

    Arguments:
        ics - Nx2 array of N initial conditions
        t - array of time points over which to simulate
        n_points - size of the high-dimensional dataset created
        linear - Boolean value. If True, high-dimensional dataset is a linear combination
        of the duffing dynamics. If False, the dataset also includes cubic modes.
        normalization - Optional 3-value array for rescaling the 3 duffing variables.
        sigma, beta, rho - Parameters of the duffing dynamics.

    Returns:
        data - Dictionary containing elements of the dataset. This includes the time points (t),
        spatial mapping (y_spatial), high-dimensional modes used to generate the full dataset
        (modes), low-dimensional duffing dynamics (z, along with 1st and 2nd derivatives dz and
        ddz), high-dimensional dataset (x, along with 1st and 2nd derivatives dx and ddx), and
        the true duffing coefficient matrix for SINDy.
    """


    n_ics = ics.shape[0]
    n_steps = t.size
    dt = t[1]-t[0]

    d = 2
    z = np.zeros((n_ics,n_steps,d))
    dz = np.zeros(z.shape)
    ddz = np.zeros(z.shape)
    for i in range(n_ics):
        z[i], dz[i], ddz[i] = simulate_duffing(ics[i], t, alpha=alpha, beta=beta, gamma=gamma, delta=delta, omega=omega)


    if normalization is not None:
        z *= normalization
        dz *= normalization
        ddz *= normalization

    n = n_points
    L = 1
    y_spatial = np.linspace(-L,L,n)

    modes = np.zeros((2*d, n))
    for i in range(2*d):
        modes[i] = legendre(i)(y_spatial)
    x1 = np.zeros((n_ics,n_steps,n))
    x2 = np.zeros((n_ics,n_steps,n))
    x3 = np.zeros((n_ics,n_steps,n))
    x4 = np.zeros((n_ics,n_steps,n))

    x = np.zeros((n_ics,n_steps,n))
    dx = np.zeros(x.shape)
    ddx = np.zeros(x.shape)
    for i in range(n_ics):
        for j in range(n_steps):
            x1[i,j] = modes[0]*z[i,j,0]
            x2[i,j] = modes[1]*z[i,j,1]
            x3[i,j] = modes[2]*z[i,j,0]**3
            x4[i,j] = modes[3]*z[i,j,1]**3

            x[i,j] = x1[i,j] + x2[i,j] + x3[i,j]
            if not linear:
                x[i,j] += x4[i,j]

            dx[i,j] = modes[0]*dz[i,j,0] + modes[1]*dz[i,j,1]
            if not linear:
                dx[i,j] += modes[3]*3*(z[i,j,0]**2)*dz[i,j,0]

            ddx[i,j] = modes[0]*ddz[i,j,0] + modes[1]*ddz[i,j,1]
            if not linear:
                ddx[i,j] += modes[3]*(6*z[i,j,0]*dz[i,j,0]**2 + 3*(z[i,j,0]**2)*ddz[i,j,0])

    # if normalization is None:
    #     sindy_coefficients = lorenz_coefficients([1,1,1], sigma=sigma, beta=beta, rho=rho)
    # else:
    #     sindy_coefficients = lorenz_coefficients(normalization, sigma=sigma, beta=beta, rho=rho)

    data = {}
    data['t'] = t
    data['y_spatial'] = y_spatial
    data['modes'] = modes
    data['x'] = x
    data['dx'] = dx
    data['ddx'] = ddx
    data['z'] = z
    data['dz'] = dz
    data['ddz'] = ddz
    # data['sindy_coefficients'] = sindy_coefficients.astype(np.float32)
    return data
