def calculate_scores(model, best_params, x_test, y_test):
    """
    Function to calculate scores for both regression models and classifiers.
    """
    y_pred = model.predict(x_test)  
    results = best_params
    regressor_condition1, regressor_condition2 = False, False

    estimator_condition = 'estimator' in dir(model)
    if estimator_condition:
        regressor_condition1 = 'Regressor' in str(type(model.estimator))
    else:
        regressor_condition2 = 'Regressor' in str(type(model))
    
    if regressor_condition1 or regressor_condition2:
        r_squared = model.score(x_test,y_test)
        mse = mean_squared_error(y_test,y_pred)
        rmse = np.sqrt(mse)

        results['rmse'] = rmse
        results['r_squared'] = r_squared
    
    else:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        sensitivity = recall_score(y_test, y_pred, average="weighted")
        
        results['accuracy'] = accuracy
        results['precision'] = precision
        results['sensitivity'] = sensitivity

    return results

def cv_score_rf(hyp_parameters):
    """
    Perform CV on RF
    """
    hyp_parameters = hyp_parameters[0]
    rf_model = RandomForestRegressor(n_estimators=int(hyp_parameters[0]),
                                 max_features=hyp_parameters[1],
                                 min_samples_split=int(hyp_parameters[2]),
                                 min_samples_leaf=int(hyp_parameters[3]),
                                 bootstrap=bool(hyp_parameters[4]))
    scores = cross_val_score(rf_model,
                             X=x_train,
                             y=y_train,
                             cv=KFold(n_splits=5))
    return np.array(scores.mean())

def cv_score_xgb(hyp_parameters):
    """
    Perform CV on XGB
    """
    hyp_parameters = hyp_parameters[0]
    xgb_model = xgb.XGBClassifier(objective="binary:logistic",
                                 learning_rate=hyp_parameters[0],
                                 max_depth=int(hyp_parameters[1]),
                                 n_estimators=int(hyp_parameters[2]),
                                 subsample=hyp_parameters[3],
                                 min_child_weight=int(hyp_parameters[4]),
                                 reg_alpha=hyp_parameters[5],
                                 reg_lambda=hyp_parameters[6],
                                njobs=-1)
    scores = cross_val_score(xgb_model,
                             X=x_train,
                             y=y_train,
                             cv=KFold(n_splits=5))
    return np.array(scores.mean())  # return average of 5-fold scores

def lines_that_start_with(string, fp):
    return [line for line in fp if line.startswith(string)]

def lines_that_contain(string, fp):
    return [line for line in fp if string in line]


def optimize_model(model, 
                   x_train, 
                   y_train, 
                   x_test, 
                   y_test,
                   opt_type, 
                   use_parallel = True,
                   cv = 5,
                   n = 3):

    # Set params for RF    
    if isinstance(model, RandomForestRegressor):
        n_estimators = np.linspace(10,150, n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(10,151)
        max_features =  np.linspace(0.05,.9, 2).tolist() if opt_type != 'randomcv' else uniform(0.05,.9)
        min_samples_split = np.linspace(2,20, n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(2,21)
        min_samples_leaf = np.linspace(1,20, n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(1,21)
        bootstrap = np.array([True, False], dtype=bool).tolist() if opt_type != 'randomcv' else [True,False]
    
        params = {
            "n_estimators": n_estimators,
            "max_features": max_features,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "bootstrap": bootstrap
        }

    # Set RFS for XGB
    else: 
        n_estimators = np.linspace(50,150, n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(50,151)
        max_depth =  np.linspace(1,10, n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(1,11)
        min_child_weight = np.linspace(1,20, n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(1,21)
        learning_rate = np.linspace(0.001,1, n).tolist() if opt_type != 'randomcv' else uniform(0.001,1)
        subsample = np.linspace(0.05,1, n).tolist() if opt_type != 'randomcv' else uniform(0.05,0.9)
        reg_lambda = np.linspace(0,5, n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(0,5)
        reg_alpha = np.arange(1,6, n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(0,5)
    
        params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "reg_lambda": reg_lambda,
            "reg_alpha:": reg_alpha
        }
    
    # Specify to use parallel processing
    n_jobs = -1 if use_parallel == True else 1
    
    # if tuning method == grid search
    if opt_type == 'gridcv':
        print("Optimizing hyperparameters with GridSearchCV...")
        grid_search = GridSearchCV(model,
                           param_grid=params,
                           cv=cv,
                           verbose=1,
                           n_jobs=n_jobs,
                           return_train_score=True)
        
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        results = calculate_scores(grid_search, best_params, x_test = x_test, y_test = y_test)
        return(results)

    # if tuning method == random CV
    elif opt_type == 'randomcv':
        print("Optimizing hyperparameters with RandomSearchCV...")
        random_search = RandomizedSearchCV(
            model,
            param_distributions=params,
            random_state=8675309,
            n_iter=10,
            cv=cv,
            verbose=1,
            n_jobs=n_jobs,
            return_train_score=True)
        
        random_search.fit(x_train, y_train)
        best_params = random_search.best_params_
        results = calculate_scores(random_search, best_params, x_test = x_test, y_test = y_test)
        return(results)
    
    # if tuning method == Bayesian optimization
    elif opt_type == "bayes":
        print("Optimizing hyperparameters with Bayesian Optimization...")
        
        # if model is RF
        if isinstance(model, RandomForestRegressor):
            hp_bounds = [{'name': 'n_estimators', 'type': 'discrete', 'domain': (min(n_estimators), max(n_estimators))}, 
            {'name': 'max_features','type': 'continuous','domain': (min(max_features), max(max_features))}, 
            {'name': 'min_samples_split','type': 'discrete','domain': (min(min_samples_split), max(min_samples_split))}, 
            {'name': 'min_samples_leaf','type': 'discrete','domain': (min(min_samples_leaf), max(min_samples_leaf))}, 
            {'name': 'bootstrap','type': 'discrete','domain': (True, False)}]
            cv_score = cv_score_rf
        
        # if model is XGB
        else:
            hp_bounds = [{'name': 'learning_rate','type': 'continuous','domain': (min(learning_rate), max(learning_rate))}, 
            {'name': 'max_depth','type': 'discrete','domain': (min(max_depth), max(max_depth))}, 
            {'name': 'n_estimators','type': 'discrete','domain': (min(n_estimators), max(n_estimators))}, 
            {'name': 'subsample','type': 'continuous','domain': (min(subsample), max(subsample))}, 
            {'name': 'min_child_weight','type': 'discrete','domain': (min(min_child_weight), max(min_child_weight))}, 
            {'name': 'reg_alpha','type': 'continuous','domain': (min(reg_alpha), max(reg_alpha))}, 
            {'name': 'reg_lambda','type': 'continuous','domain': (min(reg_lambda), max(reg_lambda))}]
            cv_score = cv_score_xgb

        # create optmizer
        optimizer = BayesianOptimization(f=cv_score,
                                         domain=hp_bounds,
                                         model_type='GP',
                                         acquisition_type='EI',
                                         acquisition_jitter=0.05,
                                         exact_feval=True,
                                         maximize=True,
                                         verbosity=True,
                                        njobs=n_jobs)

        optimizer.run_optimization(max_iter=20,verbosity=False)
        best_params = {}

        # if model is RF, convert continuous/discrete vals
        if isinstance(model, RandomForestRegressor):
            for i in range(len(hp_bounds)):
                if hp_bounds[i]['type'] == 'continuous':
                    best_params[hp_bounds[i]['name']] = optimizer.x_opt[i]
                elif hp_bounds[i]['type'] == 'discrete' and hp_bounds[i]['name'] != 'bootstrap':
                    best_params[hp_bounds[i]['name']] = int(optimizer.x_opt[i])
                else:
                    best_params[hp_bounds[i]['name']] = bool(optimizer.x_opt[i])
    
            bayopt_search = RandomForestRegressor(**best_params)
    
        # if model is XGB, do same conversion
        else:
            for i in range(len(hp_bounds)):
                if hp_bounds[i]['type'] == 'continuous':
                    best_params[hp_bounds[i]['name']] = optimizer.x_opt[i]
                else:
                    best_params[hp_bounds[i]['name']] = int(optimizer.x_opt[i])

            bayopt_search =  xgb.XGBClassifier(objective="binary:logistic", **best_params)
        
        bayopt_search.fit(x_train,y_train)
        results = calculate_scores(bayopt_search, best_params, x_test = x_test, y_test = y_test)
                
        return(results)
    
    # if tuning method is TPOT
    elif opt_type == 'tpot':
        print("Optimizing hyperparameters with TPOT...")
        
        # specify config for regressor
        if isinstance(model, RandomForestRegressor):
            tpot_config = {
                'sklearn.ensemble.RandomForestRegressor': {
                    "n_estimators": n_estimators,
                    "max_features": max_features,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "bootstrap": bootstrap
                }
            }

            tpot = TPOTRegressor(generations=5,
                                 scoring="r2",
                                 population_size=15,
                                 verbosity=2,
                                 config_dict=tpot_config,
                                 cv=cv,
                                 random_state=8675309)

            tpot.fit(x_train, y_train)
            tpot.export('tpot_rf.py')

            # process output file to get params
            with open("tpot_rf.py", "r") as fp:
                for line in lines_that_contain("Regressor", fp):
                    parse_this = line

            p = re.compile(r"[\w]+=[\w|[\d+\.\d]+")
            match_list = p.findall(parse_this)
            best_params = {}

            for match in match_list:
                key, val = match.split("=")
                best_params[key] = eval(val)

            results = calculate_scores(tpot, best_params, x_test = x_test, y_test = y_test)

        # specify config for XGB
        else:
            tpot_config = {
                'xgboost.XGBClassifier': {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate,
                    'subsample': subsample,
                    'min_child_weight': min_child_weight,
                    'reg_alpha': reg_alpha,
                    'reg_lambda': reg_lambda,
                    'nthread': [1],
                    'objective': ['binary:logistic'],
                }
            }

            tpot = TPOTClassifier(generations=5,
                     population_size=1,
                     verbosity=2,
                     config_dict=tpot_config,
                     cv=cv,
                     random_state=8675309)

            tpot.fit(x_train, y_train)
            tpot.export('tpot_xbg.py')
            
            # process output file to get params
            with open("tpot_xbg.py", "r") as fp:
                for line in lines_that_contain("XGB", fp):
                    parse_this = line

            p = re.compile(r"[\w]+=[\d+\.\d+]+")
            match_list = p.findall(parse_this)
            best_params = {}

            for match in match_list:
                key, val = match.split("=")
                best_params[key] = eval(val)

            if 'nthread' in best_params:
                del best_params['nthread']

            results = calculate_scores(tpot, best_params, x_test = x_test, y_test = y_test)
            print(results)
        return(results)

def wrapper(model, 
            x_train, 
            y_train, 
            x_test, 
            y_test,
            cv = 3, 
            n = 3):
    start_time = timeit.default_timer()
    tuning_methods = ['gridcv','randomcv','bayes','tpot']
    
    if isinstance(model, RandomForestRegressor):
        cols = ['n_estimators','max_features','min_samples_split','min_samples_leaf','bootstrap','rmse','r_squared']
    else:
        cols = ['n_estimators','max_depth','min_child_weight','learning_rate','subsample','reg_lambda','reg_alpha','accuracy','precision','sensitivity']
    
    df  = pd.DataFrame(columns = cols)
    
    print(f"CV = {cv}, n (hyperparameters) = {n}.\nParallel processing being used for all tuning methods except TPOT")
    for method in tqdm_notebook(tuning_methods):
        results = optimize_model(model, x_train, y_train, x_test, y_test, opt_type = method, cv = cv, n = n)
        df.loc[method] = results
        
    stop_time = timeit.default_timer()
        
    print(f"Done! Time elapsed: {round(stop_time - start_time)} seconds")
        
    return df