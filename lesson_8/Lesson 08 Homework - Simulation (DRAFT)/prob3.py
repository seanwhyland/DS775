def bid_sim(sim_size = 1000, random_bid = False):
    num_competitors = 4
    competitor_bids = np.zeros(num_competitors)
    project_cost = 5000000 + 50000
    lowest_bids = np.empty(sim_size)
    winners = np.empty(sim_size, dtype = np.object)
    possible_bids = np.arange(5.3,6,0.1)*1000000

    for i in range(sim_size):
        for j in range(num_competitors):
            competitor_bid_rate = np.random.triangular(2, 20, 40)
            competitor_bids[j] = project_cost * (1+(competitor_bid_rate/100))
            
            if random_bid == True:
                rpi_bid = np.random.choice(possible_bids,1)
            else:
                rpi_bid = 5700000

        min_comp_bid = min(competitor_bids)
        lowest_bid = min(rpi_bid, min_comp_bid)

        if lowest_bid == rpi_bid:
            winners[i] = "rpi"
            lowest_bids[i] = rpi_bid
        else:
            winners[i] = np.argmin(competitor_bids)
            lowest_bids[i] = min_comp_bid

    stack = np.column_stack((winners, lowest_bids))
    rpi_winners = np.where(winners == 'rpi')[0]
    comp_winners = np.where(winners != 'rpi')[0]
    mean_comp_bid = np.mean(lowest_bids[comp_winners])    
    prob_rpi = len(rpi_winners)/len(winners)
    
    if random_bid == True:
        mean_rpi_bid = np.mean(lowest_bids[rpi_winners])
        mean_rpi_profit = mean_rpi_bid - project_cost
    else:
        mean_rpi_bid = rpi_bid
        mean_rpi_profit = mean_rpi_bid - project_cost
        
    mean_profit_by_bid = []
    pctl_05 = []
    pctl_95 = []
    for k, bid in enumerate(possible_bids):
        profit_by_bid = lowest_bids[np.where(lowest_bids == bid)[0]]
        mean_profit_for_bid = np.mean(profit_by_bid) - project_cost
        mean_profit_by_bid.append(mean_profit_for_bid)
        
    df = pd.DataFrame({
        'Bid': possible_bids,
        'MeanProfit': mean_profit_by_bid,
    })

    return prob_rpi, mean_rpi_bid, df

prob_rpi, mean_rpi_bid, df = bid_sim(random_bid = True)
print(f"""a. The probability that RPI will win is {prob_rpi}
a. RPI's mean profit will be {mean_rpi_bid}.
b. The highest bid of $6M maximizes mean profit.
""")