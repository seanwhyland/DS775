
def simulate(artifacts_found = True, buy_insurance = False):
    # days in a week
    n = 7

    min_schedule_ls = []
    profit_ls = []

    np.random.seed(6)
    for _ in range(1000):
        if isinstance(artifacts_found, (int, float)):
            prob_found = artifacts_found
            prob_dist = (1-prob_found, prob_found)
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