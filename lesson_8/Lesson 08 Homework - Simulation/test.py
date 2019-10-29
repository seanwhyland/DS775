### Imports ###

from pyomo.environ import *
import pandas as pd # used for results display

def purchase_quantity(discount_pork):
    # Unfold to see the Pyomo solution with arrays of decision variables
    ### Problem Data ###

    types = ['economy','premium']
    ingredients = ['pork', 'wheat', 'starch']

    cost = dict( zip( ingredients, [4.32, 2.46, 1.86] ) )

    kg_per_sausage = 0.05

    mnpi = [[.4,.6],[0,0],[0,0]]
    min_prop_ing = { ingredients[i]:{ types[j]:mnpi[i][j] for j in range(len(types)) } for i in range(len(ingredients)) }
    mxpi = [[1,1],[1,1],[.25,.25]]
    max_prop_ing = { ingredients[i]:{ types[j]:mxpi[i][j] for j in range(len(types)) } for i in range(len(ingredients)) }

    max_ingredient = dict( zip( ingredients, [30, 20, 17] ) )
    min_ingredient = dict( zip( ingredients, [discount_pork,  0,  0] ) )

    output = []

    for _ in range(1000):
        number_each_type = dict( zip( types, [np.random.randint(325,375), np.random.randint(450,550)] ) )
        ### Pyomo Model ###

        # Concrete Model
        M = ConcreteModel(name = "Sausages")

        # Decision Variables
        M.amount = Var(ingredients, types, domain = NonNegativeReals)

        # Objective
        M.cost = Objective( expr = sum( cost[i] * sum(M.amount[i,t] for t in types) 
                                       for i in ingredients) - 1.22*(sum(M.amount["pork",t] for t in types)-discount_pork), 
                           sense = minimize )

        M.tot_sausages_ct = ConstraintList()
        for t in types:
            M.tot_sausages_ct.add( sum( M.amount[i,t] for i in ingredients ) 
                                 == kg_per_sausage * number_each_type[t] )

        M.min_prop_ct = ConstraintList()
        for i in ingredients:
            for t in types:
                M.min_prop_ct.add( M.amount[i,t] >= min_prop_ing[i][t] *
                                 sum( M.amount[k,t] for k in ingredients ) )

        M.max_prop_ct = ConstraintList()
        for i in ingredients:
            for t in types:
                M.max_prop_ct.add( M.amount[i,t] <= max_prop_ing[i][t] * 
                                 sum( M.amount[k, t] for k in ingredients ) )
                
        M.max_ingredient_ct = ConstraintList()
        for i in ingredients:
            M.max_ingredient_ct.add( sum( M.amount[ i, t] for t in types ) <= 
                                   max_ingredient[i] )
            
        M.min_ingredient_ct = ConstraintList()
        for i in ingredients:
            M.min_ingredient_ct.add( sum( M.amount[ i, t] for t in types ) >=
                                   min_ingredient[i] )

        ### Solution ###
        solver = SolverFactory('glpk')
        solver.solve(M)

        dvars = pd.DataFrame( [ [M.amount[i,t]() for t in types] for i in ingredients ],
                            index = ['Pork','Wheat','Starch'],
                            columns = ['Economy','Premium'])

        full_price_pork = sum(dvars.loc['Pork']) - discount_pork
        output.append(tuple((round(M.cost(),2), round(full_price_pork,2))))
    return output


# # Unfold to see the Pyomo solution with arrays of decision variables

# ### Imports ###

# from pyomo.environ import *
# import pandas as pd # used for results display

# ### Problem Data ###

# types = ['economy','premium']
# ingredients = ['pork', 'wheat', 'starch']

# cost = dict( zip( ingredients, [4.32, 2.46, 1.86] ) )

# kg_per_sausage = 0.05
# number_each_type = dict( zip( types, [350, 500] ) )

# mnpi = [[.4,.6],[0,0],[0,0]]
# min_prop_ing = { ingredients[i]:{ types[j]:mnpi[i][j] for j in range(len(types)) } for i in range(len(ingredients)) }
# mxpi = [[1,1],[1,1],[.25,.25]]
# max_prop_ing = { ingredients[i]:{ types[j]:mxpi[i][j] for j in range(len(types)) } for i in range(len(ingredients)) }

# max_ingredient = dict( zip( ingredients, [30, 20, 17] ) )
# min_ingredient = dict( zip( ingredients, [23,  0,  0] ) )

# ### Pyomo Model ###

# # Concrete Model
# M = ConcreteModel(name = "Sausages")

# # Decision Variables
# M.amount = Var(ingredients, types, domain = NonNegativeReals)

# # Objective
# M.cost = Objective( expr = sum( cost[i] * sum(M.amount[i,t] for t in types) 
#                                for i in ingredients) - 1.22*(sum(M.amount["pork",t] for t in types)-23), 
#                    sense = minimize )

# M.tot_sausages_ct = ConstraintList()
# for t in types:
#     M.tot_sausages_ct.add( sum( M.amount[i,t] for i in ingredients ) 
#                          == kg_per_sausage * number_each_type[t] )

# M.min_prop_ct = ConstraintList()
# for i in ingredients:
#     for t in types:
#         M.min_prop_ct.add( M.amount[i,t] >= min_prop_ing[i][t] *
#                          sum( M.amount[k,t] for k in ingredients ) )

# M.max_prop_ct = ConstraintList()
# for i in ingredients:
#     for t in types:
#         M.max_prop_ct.add( M.amount[i,t] <= max_prop_ing[i][t] * 
#                          sum( M.amount[k, t] for k in ingredients ) )
        
# M.max_ingredient_ct = ConstraintList()
# for i in ingredients:
#     M.max_ingredient_ct.add( sum( M.amount[ i, t] for t in types ) <= 
#                            max_ingredient[i] )
    
# M.min_ingredient_ct = ConstraintList()
# for i in ingredients:
#     M.min_ingredient_ct.add( sum( M.amount[ i, t] for t in types ) >=
#                            min_ingredient[i] )

# ### Solution ###
# solver = SolverFactory('glpk')
# solver.solve(M)

# ### Display ###

# import babel.numbers as numbers  # needed to display as currency
# print("Total Cost = ",
#       numbers.format_currency(M.cost(), 'USD', locale='en_US'))

# # put amounts in dataframe for nicer display
# import pandas as pd
# dvars = pd.DataFrame( [ [M.amount[i,t]() for t in types] for i in ingredients ],
#                     index = ['Pork','Wheat','Starch'],
#                     columns = ['Economy','Premium'])
# print("Kilograms of each ingredient in each type of sausage:")
# dvars