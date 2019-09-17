# import numpy as np
# sources = (1,2,3)
# markets = (1,2,3,4,5)
# routes_rail = [(s,m) for s in sources for m in markets]
# routes_ship = routes_rail.copy()
# routes_ship.remove((1,4))
# routes_ship.remove((3,1))
# supply_dict = {1: 15, 2: 20, 3:15}
# demand_dict = {1: 11, 2: 12, 3: 9, 4: 10, 5: 8}
# costs_rail = np.multiply((61,72,45,55,66,69,78,60,49,56,59,66,63,61,47),1000).tolist()
# # costs_ship = np.multiply((31,38,24,35,36,43,28,24,31,33,36,32,26),1000).tolist()
# costs_ship_bm = np.multiply((31,38,24,1000,35,36,43,28,24,31,1000,33,36,32,26),1000).tolist()
# # costs_ship_inv = np.multiply((275, 303, 238, 285, 293, 318, 270, 250, 265, 283, 275, 268, 240),100).tolist()
# costs_ship_inv_bm = np.multiply((275, 303, 238, 10000, 285, 293, 318, 270, 250, 265, 10000, 283, 275, 268, 240),100).tolist()
# route_costs_rail = {(s,m):costs_rail[i] for i, (s,m) in enumerate(routes_rail)}
# route_costs_ship = {(s,m):costs_ship[i]+costs_ship_inv[i] for i, (s,m) in enumerate(routes_ship)}
# route_costs_ship_bm = {(s,m):costs_ship_bm[i]+costs_ship_inv_bm[i] for i, (s,m) in enumerate(routes_ship)}
# route_costs_all = {route:min((cost_rail, cost_ship)) for route, (cost_rail, cost_ship) in zip(routes_rail,zip(route_costs_rail.values(),route_costs_ship_bm.values()))}


# # set vars
# model.transp = Var(routes_rail, domain=NonNegativeReals)

# # objective function
# model.total_cost = Objective(expr=sum(route_costs_rail[s,m] * model.transp[s,m] for (s,m) in routes_rail), sense=minimize)

# # supply constraint
# model.supply_ct = ConstraintList()
# for s in sources:
#     model.supply_ct.add(sum(model.transp[s,m] for m in markets if (s,m) in routes_rail) <= supply_dict[s] )

# # demand constraint    
# model.demand_ct = ConstraintList()
# for m in markets:
#     model.demand_ct.add(sum(model.transp[s,m] for s in sources if (s,m) in routes_rail) == demand_dict[m] )

# # solve and display
# solver = SolverFactory('glpk')
# solver.solve(model)

# # # convert model.hrs into a Pandas data frame for nicer display
# import pandas as pd
# table = pd.DataFrame(0, index=sources, columns=markets)
# for (s, m) in routes_rail:
#     table.loc[s, m] = model.transp[s, m].value

# # display
# import babel.numbers as numbers  # needed to display as currency
# print("Option 1: The minimum total transportation cost for rail = ",
#       numbers.format_currency(model.total_cost(), 'USD', locale='en_US'))

# from IPython.display import display
# display(table)



import numpy as np
import pprint as pp


sources = (1,2,3)
markets = (1,2,3,4,5)
routes_rail = [(s,m) for s in sources for m in markets]
routes_ship = routes_rail.copy()
# routes_ship.remove((1,4))
# routes_ship.remove((3,1))
supply_dict = {1: 15, 2: 20, 3:15}
demand_dict = {1: 11, 2: 12, 3: 9, 4: 10, 5: 8}
costs_rail = np.multiply((61,72,45,55,66,69,78,60,49,56,59,66,63,61,47),1000).tolist()
costs_ship = np.multiply((31,38,24,1000,35,36,43,28,24,31,1000,33,36,32,26),1000).tolist()
costs_ship_inv = np.multiply((275, 303, 238, 10000, 285, 293, 318, 270, 250, 265, 10000, 283, 275, 268, 240),100).tolist()
route_costs_rail = {(s,m):costs_rail[i] for i, (s,m) in enumerate(routes_rail)}
route_costs_ship = {(s,m):costs_ship[i]+costs_ship_inv[i] for i, (s,m) in enumerate(routes_ship)}
route_costs_all = {route:min((cost_rail, cost_ship)) for route, (cost_rail, cost_ship) in zip(routes_rail,zip(route_costs_rail.values(),route_costs_ship.values()))}

pp.pprint(route_costs_rail)
pp.pprint(route_costs_ship)

