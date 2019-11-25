def simulate(artifacts_found = True, buy_insurance = False):
    # days in a week
    n = 7

    min_schedule_ls = []
    profit_ls = []

    np.random.seed(6)
    
    if isinstance(artifacts_found, (int, float)):
        prob_found = artifacts_found
        prob_dist = (1-prob_found, prob_found)

    for j in range(1000):
        if prob_dist:
            artifacts_found = np.random.choice([0,1], 1, p=[*prob_dist])

        if artifacts_found == 1 or artifacts_found is True:
            time_dist = int(np.random.triangular(7/n, 15/n, 365/n))
        else:
            time_dist = int(np.random.triangular(7/n, 14/n, 21/n))

        task_duration_dict = {
            'excavate': time_dist,
            'lay_foundation': int(np.random.triangular(14/n, 21/n, 56/n)),
            'rough_wall': int(np.random.triangular(42/n, 63/n, 126/n)),
            'roof': int(np.random.triangular(28/n, 35/n, 70/n)),
            'exterior_plumbing': int(np.random.triangular(7/n, 28/n, 35/n)),
            'interior_plumbing': int(np.random.triangular(28/n, 35/n, 70/n)),
            'exterior_siding': int(np.random.triangular(35/n, 42/n, 77/n)),
            'exterior_painting': int(np.random.triangular(35/n, 56/n, 119/n)),
            'electrical_work': int(np.random.triangular(21/n, 49/n, 63/n)),
            'wallboard': int(np.random.triangular(21/n, 63/n, 63/n)),
            'flooring': int(np.random.triangular(21/n, 28/n, 28/n)),
            'interior_painting': int(np.random.triangular(7/n, 35/n, 49/n)),
            'exterior_fixtures': int(np.random.triangular(7/n, 14/n, 21/n)),
            'interior_fixtures': int(np.random.triangular(35/n, 35/n, 63/n))
        }
        task_names = list(task_duration_dict.keys())
        num_tasks = len(task_names)
        durations = list(task_duration_dict.values())

        task_name_to_number_dict = dict(zip(task_names, np.arange(0, num_tasks)))

        horizon = sum(task_duration_dict.values())

        from ortools.sat.python import cp_model
        model = cp_model.CpModel()

        start_vars = [
            model.NewIntVar(0, horizon, name=f'start_{t}') for t in task_names
        ]
        end_vars = [model.NewIntVar(0, horizon, name=f'end_{t}') for t in task_names]

        # the `NewIntervalVar` are both variables and constraints, the internally enforce that start + duration = end
        intervals = [
            model.NewIntervalVar(start_vars[i],
                                 durations[i],
                                 end_vars[i],
                                 name=f'interval_{task_names[i]}')
            for i in range(num_tasks)
        ]

        # precedence constraints
        for before in list(precedence_dict.keys()):
            for after in precedence_dict[before]:
                before_index = task_name_to_number_dict[before]
                after_index = task_name_to_number_dict[after]
                model.Add(end_vars[before_index] <= start_vars[after_index])

        obj_var = model.NewIntVar(0, horizon, 'largest_end_time')
        model.AddMaxEquality(obj_var, end_vars)
        model.Minimize(obj_var)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        
        # optimal schedule in days
        osl_days = solver.ObjectiveValue()*n
        
        # append optimal schedule to list
        min_schedule_ls.append(osl_days)
        
        import math
        
        base, bonus, penalty = 5400000, 0, 0

        # define insurance costs
        if buy_insurance == False:
            insurance_cost = 0
        else:
            insurance_cost = 500000

        # define artifact costs 
        if artifacts_found == 1 or artifacts_found == True:
            if buy_insurance == False:
                artifact_cost = np.random.exponential(scale=100000)
            else:
                artifact_cost = 0

        else:
            artifact_cost = 0

        # define bonus and penalty for days over deadline
        if math.ceil(osl_days) < 280:
            bonus = 150000
            
        elif math.ceil(osl_days) > 329:
            days_over = int(math.ceil(osl_days)) - 329
            penalty = days_over * 25000
        
        # calculate profit
        profit = base + bonus - (penalty + artifact_cost + insurance_cost)
        profit_ls.append(profit)

    return profit_ls, min_schedule_ls


def return_stats(profit_ls, min_schedule_ls, show_summary = True):
    profit_ls = np.array(profit_ls)
    min_schedule_ls = np.array(min_schedule_ls)

    # calculate summary stats
    mean_profit = int(np.mean(profit_ls))
    less_than_280 = len(np.where(min_schedule_ls < 280)[0])/len(min_schedule_ls)
    between_280_and_329 = len(np.intersect1d(np.where(min_schedule_ls >= 280)[0],np.where(min_schedule_ls <= 329)[0]))/len(min_schedule_ls)
    over_329 = len(np.where(min_schedule_ls > 329)[0])/len(min_schedule_ls)

    # print summary stats else return them
    if show_summary is True:
        print(f"""Summary Stats:
        mean profit: ${mean_profit:,.2f}
        prob less than 280 days: {round(less_than_280*100,2)}%
        prob between 280 and 329 days: {round(between_280_and_329*100,2)}%
        prob over 329 days: {round(over_329*100,2)}%
        prob sum: {(less_than_280 + between_280_and_329 + over_329)*100}%""")

    else:
        return mean_profit

def show_payoff_table(artifacts_found = .30):
    import pandas as pd

    # define states, alternatives, and prior probs
    alternatives = {'Buy_Insurance': True,'No_Insurance': False}
    states =  {'Artifacts': True, 'No_Artifacts': False}
    df  = pd.DataFrame(columns = list(states.keys()), index=list(alternatives.keys()))
    prior_probs = [artifacts_found, 1-artifacts_found]

    # populate payoff table
    for alt_name, alt_val in alternatives.items():
        for state_name, state_value in states.items():
            profit_ls, min_schedule_ls = simulate(artifacts_found = state_value, buy_insurance = alt_val)
            mean_profit = return_stats(profit_ls, min_schedule_ls, show_summary = False)
            df.loc[alt_name][state_name] = round(mean_profit/1000000,1)

    # df.loc[:,['Artifacts','No_Artifacts']].apply(lambda x: x/1000000).style.format('${0:,.2f}')

    return df, prior_probs


def bayes_calc(prior_probs, df):
    # create arrays of alternatives and prior probs
    alt_states = np.array([df.loc["Buy_Insurance"].tolist(),df.loc["No_Insurance"].tolist()])
    prior_probs = np.array(prior_probs)
    expected_payoffs = {}

    # calculate expected payoffs using Bayes' decision rule
    for i, alt in enumerate(alt_states):
        ep = sum(prior_probs * np.array(alt))
        expected_payoffs[df.index[i]] = ep

    # get maximum payoff and best alternative
    best_alt = max(expected_payoffs, key=expected_payoffs.get)
    max_val = expected_payoffs[best_alt]
    return best_alt, max_val

