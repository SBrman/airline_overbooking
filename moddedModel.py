import numpy as np
from poibin.poibin import PoiBin
from matplotlib import pyplot as plt
from model_loader import *


np.random.seed(101)


# Constants
FLIGHT_DATA_ALL = np.genfromtxt('./data/final_flight_test_cc.csv', delimiter=',')
FLIGHT_DATA = FLIGHT_DATA_ALL[1:, 2:]
FLIGHTS = FLIGHT_DATA[:, -1]
MODEL_PATH = './final_models/Neural_network3d_51_179_t_1670230822.552445.pt'
# MODEL_PATH = './final_models/Logistic_regression_51_179_t_1670226651.3897946.pt'
# MODEL_PATH = './final_models/NNLR_51_179_t_1670228774.5908124.pt'

NETWORK = NN4d(INPUT_SIZE, OUTPUT_SIZE).to(device)
MODEL = Model(NETWORK, MODEL_PATH) 


def binomial_pmf(p):
    """
    Usage:
    >>> total_booked_tickets = 100
    >>> average_show_up_probability = 0.93
    >>> binPMF = binomial_pmf(average_show_up_probability)
    >>> binPMF(p=total_booked_tickets, r=80)    # Probability of exactly 80 people showing up
    1.2875022596782589e-05
    >>> binPMF(p=total_booked_tickets, r=90)    # Probability of exactly 80 people showing up
    0.07124437842927556
    """
    def probability_of_exactly_r_passenger_showing_up(n, r):
        nCr = np.math.factorial(n) / (np.math.factorial(r) * np.math.factorial(n-r))
        return nCr * (p**r) * (1 - p)**(n - r)

    return probability_of_exactly_r_passenger_showing_up


def poisson_binomial_pmf(p):
    """
    Usage:
    p is a list or numpy array.
    Other than that the usage is the same as binomial_pmf function.
    """
    return lambda n, k: PoiBin(p[:n]).pmf(k)


def get_overbook_number(probabilities, max_overbook, penalty, method='binomial'):
    """
    probabilities: float or list
    max_overbook: int
    penalty: float
    method: ['binomial', 'poisson_binomial']

    If the method is binomial, the expected input for probabilities is a floating point number.
    If the method is poisson_binomial, the expected input for probabilities is a list of
    individual passengers show up probabilities.

    returns two lists: overbook numbers and revenues
    """
    if method == 'binomial':
        assert isinstance(probabilities, float), 'Probabilities must be the average probability.'
    elif method == 'poisson_binomial':
        assert type(probabilities) in {list, np.ndarray}, 'Probabilities must be the list of individual probabilities.'
    else:
        raise NotImplementedError(f'{method} is not implemented.')
    
    x, y = [], []
    sum1 = 0
    p = probabilities
    
    pmf = binomial_pmf(p) if method == 'binomial' else poisson_binomial_pmf(p)

    for overbook in range(max_overbook + 1):

        N = 186 + overbook  #186 is the capacity
        
        for k in range(186, N+1):
            sum1 = sum1 + (k - 186) * pmf(N, k)
        
        revenue = overbook - penalty * sum1
        
        if len(y) > 10 and all([i <= 0 for i in y[-2:]]): break
        
        x.append(overbook)
        y.append(revenue)
        
    return x[:-1], y[:-1]


def plot_results(x, y, px, py, flight, plot_title=None):
    
    plt.figure(figsize=(16, 9))
    
    plt.plot(x, y, label='Base Model')
    plt.scatter(y.index(max(y)), max(y), label='Base model revenue according to base model recommendations')

    base_rec = y.index(max(y))
    plt.scatter(base_rec, py[base_rec], label='Actual revenue (assuming ML model is more accurate) with base model overbooking recommendation')

    plt.plot(px, py, label='Modified Model')
    plt.scatter(py.index(max(py)), max(py), label='Actual revenue (assuming ML model is more accurate) with ML model overbooking recommendation')

    # Just a horizontal line to show profit vs loss
    plt.plot([0, max(max(x), max(px))+1], [0, 0], label='Break even')

    plt.xlabel('Overbooking Recommendation')
    plt.ylabel('Profit')
    
    if plot_title is not None:
        plt.suptitle(plot_title)
    else:
        plt.suptitle(f"Flight {int(flight)} overbooking recommendations.")
    
    plt.legend(loc=0)
    
    plt.savefig(f'./figs/NN4d_{int(flight)}.png')
    plt.close()
    # plt.show()


def get_individual_probabilities(flight_number):
    """
    The ML model will give us the individual probabilities.
    Usage:
    ------
    >>> get_individual_probabilities(flight_number=411)
    [0.6961069055169528, 0.7381605669414192, 0.732393849034117, ... ] 
    """
    passenger_indeces = np.where(FLIGHTS == flight_number)[0]
    
    if len(passenger_indeces) < 300:
        artificial_indeces = np.random.choice(passenger_indeces, size=(300-len(passenger_indeces)))
        indeces = np.array([*passenger_indeces, *artificial_indeces])
    else:
        indeces = np.array(passenger_indeces)
        
    np.random.shuffle(indeces)
    
    passenger_features = FLIGHT_DATA[indeces][:, :-1]
    passenger_features = torch.tensor(passenger_features).to(device)

    probabilities = MODEL(passenger_features)
    probabilities = torch.flatten(probabilities).tolist()
    
    return probabilities
