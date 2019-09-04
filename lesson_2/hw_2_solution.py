# unfold to see Pyomo solution
from pyomo.environ import *

days = ["mon","tues","weds","thurs","fri"]
max_hours = dict(zip(days, [14,14,14,14,14]))

students = ['kc', 'dh', 'hb', 'sc', 'ks', 'nk']
wages = dict(zip(students, [25, 26, 24, 23, 28, 30]))

supply = [[6,0,6,0,6], [0,6,0,6,0], [4,8,4,0,4], [5,5,5,0,5], [3,0,3,8,0], [0,0,0,6,2]]

scheduled_hours = {}

for i, student in enumerate(students):
    student_dict = {}
    
    for j, day in enumerate(days):
        student_dict[day] = supply[i][j]
        
    schedule_cost[student] = student_dict


model = ConcreteModel()

model.schedule = Var(students, days, domain=NonNegativeReals)

model.total_cost = Objective(expr = sum(wages[student] * model.schedule[student, day]
                                      for student in students for day in days),
                             sense=minimize)

model.display()

# model.supply_ct = ConstraintList()

# for student in students:
#     model.supply_ct.add(
#         sum(model.schedule[student, day] for day in days) <= schedule_cost[student][day])
    
# for student in students:
#     model.supply_ct.add(
#         sum(model.schedule[student] for day in days) <= schedule_cost[student][day])

# model.demand_ct = ConstraintList()

# # for day in days:
# #     model.demand_ct.add(
# #         sum(model.schedule[student, day] for student in students) == max_hours[day])

# # solve and display
# solver = SolverFactory('glpk')
# solver.solve(model)

# # display solution
# import babel.numbers as numbers  # needed to display as currency
# print(model.total_cost())

# import pandas as pd
# dvars = pd.DataFrame([[model.schedule[student, day]() for day in days]
#                       for student in students],
#                      index=students,
#                      columns=days)
# # print("Number to ship from each factory to each customer:")
# dvars