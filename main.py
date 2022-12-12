from moddedModel import *

# Base model probability confidence
BC = 1
# ML model probability confidence
MLC = 0.67


def generate_overbooking(flight_number, avg_show_up_prob, penalty, capacity, plot_title=None):
    """
    returns a tuple of the following:
    ---------------------------------
    overbooking recommendation from the base 
    overbooking recommendation from the ML model
    the revenue from the ML model 
    the revenue according to the ML model if overbooking is done according to the base model
    the revenue from the base model
    
    rtype: tuple[int, int, float, float, float]
    
    Calling this function also creates the plot ./figs/NN4d_{flight_number}.png
    
    Usage:
    ------
    >>> generate_overbooking(3515, p, penalty, capacity)
     Recommended overbooking level with Base model:  11
     Recommended overbooking level with ML model:  14
     (11, 14, 12.521100849230443, 10.786791797839605, 9.284591181312527)

    """
    p = avg_show_up_prob
    
    # ps are the different individual probabilities of passengers showing up.
    ps = get_individual_probabilities(flight_number)
    total_bookings_requests_so_far = len(ps)
    maxoverbook = total_bookings_requests_so_far - capacity

    # Base Model
    x, y = get_overbook_number(probabilities=p, max_overbook=maxoverbook, penalty=penalty, method='binomial')
    base_rev, base_rec = max((y_i, i) for i, y_i in enumerate(y))
    print("Recommended overbooking level with Base model: ", base_rec)

    # ML Model
    # Weighted averages of the probabilities of Base model and ML model
    ps = [min(1, p*(BC/(BC + MLC)) + p1*(MLC/(BC + MLC))) for p1 in ps]
    
    px, py = get_overbook_number(ps, maxoverbook, penalty, method='poisson_binomial')
    modded_rev, modded_rec = max((y_i, i) for i, y_i in enumerate(py))
    print("Recommended overbooking level with ML model: ", modded_rec)

    plot_results(x, y, px, py, flight_number, plot_title)
    
    base_actual_rev = py[base_rec]
    return base_rec, modded_rec, modded_rev, base_actual_rev, base_rev


def main():
    flight_records = {}
    
    for flight_number in np.unique(FLIGHTS).astype(int):
        
        base_rec, ml_rec, mr, ar, br = generate_overbooking(flight_number, p, penalty, capacity, plot_title=None)

        flight_records[flight_number] = {
            "modded_revenue": mr, "base_actual_revenue": ar, "base_revenue": br, 
            "Revenue increase": mr - ar, "Base model overbooking recommendation": base_rec,
            "ML model overbooking recommendation": ml_rec
        }
        
    rev_increase = {fn: flight_data['Revenue increase'] for fn, flight_data in flight_records.items()}
    plt.bar(list(range(len(rev_increase))), list(rev_increase.values()))
    plt.xticks(list(range(len(rev_increase))), list(rev_increase), rotation='vertical')
    plt.xlabel('Flight number')
    plt.ylabel(r'Revenue change')
    plt.tight_layout()
    plt.savefig('./figs/rev_change.png')
    

if __name__ == '__main__':
    
    p = 0.93
    penalty = 2
    capacity = 186
    
##    main()

    # Usage example:
    # base_rec, ml_rec, mr, ar, br = generate_overbooking(flight_number, p, penalty, capacity, plot_title=None)
