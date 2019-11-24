import numpy as np

activity_map = {
    'A': 'excavate',
    'B': 'lay_foundation',
    'C': 'rough_wall',
    'D': 'roof',
    'E': 'exterior_plumbing',
    'F': 'interior_plumbing',
    'G': 'exterior_siding',
    'H': 'exterior_painting',
    'I': 'electrical_work',
    'J': 'wallboard',
    'K': 'flooring',
    'L': 'interior_painting',
    'M': 'exterior_fixtures',
    'N': 'interior_fixtures'
}

activities = {
    'A': '',
    'B': 'A',
    'C': 'B',
    'D': 'C',
    'E': 'C',
    'F': 'E',
    'G': 'D',
    'H': ['E','G'],
    'I': 'C',
    'J': ['F','I'],
    'K': 'J',
    'L': 'J',
    'M': 'H',
    'N': ['K','L']
}

precedence_dict = {}

for k, v in activities.items():
    for k2, v2 in activities.items():
        if isinstance(v2, list):
            if k in v2 or k == v2:
                if activity_map[k] not in precedence_dict:
                    precedence_dict[activity_map[k]] = [activity_map[k2]]
                else:
                    precedence_dict[activity_map[k]].append(activity_map[k2])
        else:
            if k == v2:
                if activity_map[k] not in precedence_dict:
                    precedence_dict[activity_map[k]] = [activity_map[k2]]
                else:
                    precedence_dict[activity_map[k]].append(activity_map[k2])

# days in a week
n = 7

min_schedule_ls = []
profit_ls = []


np.random.seed(6)
for _ in range(1000):
    task_duration_dict = {
        'excavate': int(np.random.triangular(7/n, 14/n, 21/n)),
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
    
    # calculate profit
    import math
    
    base, bonus, penalty = 5400000, 0, 0
    
    if math.ceil(osl_days) < 280:
        bonus = 150000
        
    elif math.ceil(osl_days) > 329:
        days_over = int(math.ceil(osl_days)) - 329
        penalty = days_over * 25000
    
    profit = base + bonus - penalty
    profit_ls.append(profit)
    
def plot_stats(profit_ls, min_schedule_ls):
    profit_ls = np.array(profit_ls)
    min_schedule_ls = np.array(min_schedule_ls)

    mean_profit = int(np.mean(profit_ls))
    less_than_280 = len(np.where(min_schedule_ls < 280)[0])/len(min_schedule_ls)
    between_280_and_329 = len(np.intersect1d(np.where(min_schedule_ls >= 280)[0],np.where(min_schedule_ls <= 329)[0]))/len(min_schedule_ls)
    over_329 = len(np.where(min_schedule_ls > 329)[0])/len(min_schedule_ls)

    print(f"""Summary Stats:
    mean profit: {mean_profit}
    prob less than 280 days: {round(less_than_280*100,2)}%
    prob between 280 and 329 days: {round(between_280_and_329*100,2)}%
    prob over 329 days: {round(over_329*100,2)}%
    prob sum: {(less_than_280 + between_280_and_329 + over_329)*100}%""")    