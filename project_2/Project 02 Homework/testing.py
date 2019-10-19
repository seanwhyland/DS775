def calculate_scores(model, best_params, x_test = x_test, y_test = y_test):
    r_squared = model.score(x_test,y_test)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)

    results = best_params
    results['rmse'] = rmse
    results['r_squared'] = r_squared

    return results

def cv_score(hyp_parameters):
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

def lines_that_start_with(string, fp):
    return [line for line in fp if line.startswith(string)]

def optimize_model(model, 
                   opt_type, 
                   x_train = x_train, 
                   y_train = y_train, 
                   x_test = x_test, 
                   y_test = y_test, 
                   use_parallel = True,
                   cv = 5):
    
    n = 3 if opt_type != 'randomcv' else 5
    
    if isinstance(model, RandomForestRegressor):
        n_estimators = np.arange(10,170,30).tolist() if opt_type != 'randomcv' else randint(10,151)
        max_features =  np.linspace(0.05,1,n).tolist() if opt_type != 'randomcv' else uniform(0.05,1)
        min_samples_split = np.linspace(2,20,n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(2,21)
        min_samples_leaf = np.linspace(1,20,n, dtype=np.int).tolist() if opt_type != 'randomcv' else randint(1,21)
        bootstrap = np.array([True, False], dtype=bool).tolist() if opt_type != 'randomcv' else [True,False]
	
	    params = {
	        "n_estimators": n_estimators,
	        "max_features": max_features,
	        "min_samples_split": min_samples_split,
	        "min_samples_leaf": min_samples_leaf,
	        "bootstrap": bootstrap
	    }
    else: 
        n_estimators = np.arange(50,151,25).tolist() if opt_type != 'randomcv' else randint(50,151)
        max_depth =  np.arange(1,11,2).tolist() if opt_type != 'randomcv' else randint(1,11)
        min_child_weight = np.arange(1,22,6).tolist() if opt_type != 'randomcv' else randint(1,21)
        learning_rate = np.arange(50,151,25).tolist() if opt_type != 'randomcv' else uniform(0.001,1)
        subsample = np.arange(0.05,1.2,.25).tolist() if opt_type != 'randomcv' else uniform(0.05,1)
        reg_lambda = np.arange(1,6,2).tolist() if opt_type != 'randomcv' else randint(0,5)
        reg_alpha = np.arange(1,6,2).tolist() if opt_type != 'randomcv' else randint(0,5)
    
	    params = {
	        "learning_rate": learning_rate,
	        "max_depth": max_depth,
	        "n_estimators": n_estimators,
	        "subsample": subsample,
	        "min_child_weight": min_child_weight,
	        "reg_lambda": reg_lambda,
	        "reg_alpha:": reg_alpha
	    }
    
    n_jobs = -1 if use_parallel == True else 1
    
    if opt_type == 'gridcv':
        print("Optimizing hyperparameters with GridSearchCV...")
        grid_search = GridSearchCV(model,
                           param_grid=params,
                           cv=cv,
                           verbose=0,
                           n_jobs=n_jobs,
                           return_train_score=True)
        
        grid_search.fit(x_train, y_train)
        best_params = grid_search.best_params_
        results = calculate_scores(grid_search, best_params)
        return(results)

    elif opt_type == 'randomcv':
        print("Optimizing hyperparameters with RandomSearchCV...")
        random_search = RandomizedSearchCV(
            model,
            param_distributions=params,
            random_state=8675309,
            n_iter=25,
            cv=cv,
            verbose=0,
            n_jobs=n_jobs,
            return_train_score=True)
        
        random_search.fit(x_train, y_train)
        best_params = random_search.best_params_
        results = calculate_scores(random_search, best_params)
        return(results)
    
    elif opt_type == "bayes":
        print("Optimizing hyperparameters with Bayesian Optimization...")
        if isinstance(model, RandomForestRegressor):
	        hp_bounds = [{'name': 'n_estimators', 'type': 'discrete', 'domain': (min(n_estimators), max(n_estimators))}, 
	        {'name': 'max_features','type': 'continuous','domain': (min(max_features), max(max_features))}, 
	        {'name': 'min_samples_split','type': 'discrete','domain': (min(min_samples_split), max(min_samples_split))}, 
	        {'name': 'min_samples_leaf','type': 'discrete','domain': (min(min_samples_leaf), max(min_samples_leaf))}, 
	        {'name': 'bootstrap','type': 'discrete','domain': (True, False)}]
	        cv_score = cv_score_rf
	    
	    else:
			hp_bounds = [{'name': 'learning_rate','type': 'continuous','domain': (min(learning_rate), max(learning_rate))}, 
			{'name': 'max_depth','type': 'discrete','domain': (min(max_depth), max(max_depth))}, 
			{'name': 'n_estimators','type': 'discrete','domain': (min(n_estimators), max(n_estimators))}, 
			{'name': 'subsample','type': 'continuous','domain': (min(subsample), max(subsample))}, 
			{'name': 'min_child_weight','type': 'discrete','domain': (min(min_child_weight), max(min_child_weight))}, 
			{'name': 'reg_alpha','type': 'continuous','domain': (min(reg_alpha), max(reg_alpha))}, 
			{'name': 'reg_lambda','type': 'continuous','domain': (min(reg_lambda), max(reg_lambda))}]
			cv_score = cv_score_xgb


        optimizer = BayesianOptimization(f=cv_score,
                                         domain=hp_bounds,
                                         model_type='GP',
                                         acquisition_type='EI',
                                         acquisition_jitter=0.05,
                                         exact_feval=True,
                                         maximize=True,
                                         verbosity=False,
                                        njobs=n_jobs)

        optimizer.run_optimization(max_iter=20,verbosity=False)
        
        best_params = {}

        if isinstance(model, RandomForestRegressor):
	        for i in range(len(hp_bounds)):
	            if hp_bounds[i]['type'] == 'continuous':
	                best_params[hp_bounds[i]['name']] = optimizer.x_opt[i]
	            elif hp_bounds[i]['type'] == 'discrete' and hp_bounds[i]['name'] != 'bootstrap':
	                best_params[hp_bounds[i]['name']] = int(optimizer.x_opt[i])
	            else:
	                best_params[hp_bounds[i]['name']] = bool(optimizer.x_opt[i])
	
	        bayopt_search = RandomForestRegressor(**best_params)
	        bayopt_search.fit(x_train,y_train)
	        results = calculate_scores(bayopt_search, best_params)
	    else:
			for i in range(len(hp_bounds)):
			    if hp_bounds[i]['type'] == 'continuous':
			        best_params[hp_bounds[i]['name']] = optimizer.x_opt[i]
			    else:
			        best_params[hp_bounds[i]['name']] = int(optimizer.x_opt[i])

	        bayopt_search =  xgb.XGBClassifier(objective="binary:logistic", **best_params)
	        bayopt_search.fit(x_train,y_train)
	        results = calculate_scores(bayopt_search, best_params)
                
        return(results)
    
    elif opt_type == 'tpot':
        print("Optimizing hyperparameters with TPOT...")
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
                             population_size=20,
                             verbosity=0,
                             config_dict=tpot_config,
                             cv=3,
                             random_state=8675309)
        tpot.fit(x_train, y_train)
        tpot.export('tpot_rf.py')
        
        with open("tpot_rf.py", "r") as fp:
            for line in lines_that_start_with("exported_pipeline = ", fp):
                parse_this = line

        p = re.compile(r"[\w]+=[\w|[\d+\.\d]+")
        match_list = p.findall(parse_this)
        best_params = {}

        for match in match_list:
            key, val = match.split("=")
            best_params[key] = eval(val)
                
        results = calculate_scores(tpot, best_params)
        return(results)

def wrapper(model):
    start_time = timeit.default_timer()
    tuning_methods = ['gridcv','randomcv','bayes','tpot']
    cols = ['n_estimators','max_features','min_samples_split','min_samples_leaf','bootstrap','rmse','r_squared']
    df  = pd.DataFrame(columns = cols)
    
    print("Parallel processing being used for all tuning methods except TPOT")
    for method in tqdm_notebook(tuning_methods):
        results = optimize_model(model, opt_type = method)
        df.loc[method] = results
        
    stop_time = timeit.default_timer()
        
    print(f"Done! Time elapsed: {round(stop_time - start_time)} seconds")
        
    return df