import gurobipy as gp 
import time
from gurobipy import GRB
import networkx as nx
from collections import defaultdict

from helper import Service, hhmm2mins, mins2hhmm, fetch_data, draw_graph_with_edges, node_legal, no_overlap, create_duty_graph, extract_nodes, generate_paths, roster_statistics, get_bad_paths, get_lazy_constraints, generate_initial_feasible_duties_random_from_services, restricted_linear_program, generate_new_column, generate_new_column_2, restricted_linear_program_for_heuristic

def simple_mpc(graph, service_dict, show_logs = True, show_duties = False, show_roster_stats = False):

    model = gp.Model("MPC")

    incoming_var = defaultdict(list)
    outgoing_var = defaultdict(list) 
    edge_vars = {} #xij - binary

    incoming_adj_list = nx.to_dict_of_lists(graph.reverse())

    #Decision Variables
    for (i,j) in graph.edges():
        edge_vars[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        incoming_var[j].append(edge_vars[i,j])
        outgoing_var[i].append(edge_vars[i,j])


    #Objective 
    model.setObjective(gp.quicksum(edge_vars[i,-1] for i in graph.nodes if i not in [-1, -2]), GRB.MINIMIZE)


    #Constraints - Flow conservation
    flow_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(incoming_var[i])== 1,name=f"Service_flow_{i}") #gp.quicksum(outgoing_var[i]
            flow_constraints.append(constr)



    cover_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1,name=f"cover_{i}")
            cover_constraints.append(constr)

    if not show_logs:
        print("\nNumber of decision variables: ", len(edge_vars))
        print("Number of flow constraints: ", len(flow_constraints))
        print("Number of cover constraints: ", len(cover_constraints))
        model.setParam('OutputFlag', 0)
    model.optimize()

    paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
    if show_roster_stats:
        roster_statistics(paths, service_dict) 
    return paths, len(paths)

def mpc_duration_constr(graph, service_dict, show_logs = True, max_duty_duration=6*60, time_limit = 60, show_duties = False, show_roster_stats = False):

    model = gp.Model("MPC")
    model.setParam('OutputFlag', 0)
    if time_limit:
        model.setParam('TimeLimit', time_limit)

    incoming_var = defaultdict(list)
    outgoing_var = defaultdict(list)
    incoming_relation_var = defaultdict(list) 
    edge_vars = {} #xij - binary
    edge_cumu = {} #zij -continuous

    incoming_adj_list = nx.to_dict_of_lists(graph.reverse())
    # outgoing_adj_list = nx.to_dict_of_lists(graph)

    #Decision Variables
    for (i,j) in graph.edges():
        # print(i,j)
        edge_vars[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        incoming_var[j].append(edge_vars[i,j])
        outgoing_var[i].append(edge_vars[i,j])
        edge_cumu[i,j] = model.addVar(vtype=GRB.CONTINUOUS, lb = 0, name=f"z_{i}_{j}")
        incoming_relation_var[j].append(edge_cumu[i,j])

    path_duration = {}    
    for node in graph.nodes():
        path_duration[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{node}")

    # path_duration[-1] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{-1}")
    # path_duration[-2] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{-2}")

    #Objective 
    model.setObjective(gp.quicksum(edge_vars[i,-1] for i in graph.nodes if i not in [-1, -2]), GRB.MINIMIZE)
    # print(incoming_var)

    #Constraints - Flow conservation
    flow_constraints = []
    z_y_relation_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(incoming_var[i])== 1,name=f"Service_flow_{i}") #gp.quicksum(outgoing_var[i]
            flow_constraints.append(constr)

            #z_y relationship constraints -- new model
            constr2 = model.addConstr(path_duration[i] >= service_dict[i].serv_dur + gp.quicksum(edge_cumu[j,i] for j in incoming_adj_list[i]),name=f"relation_{i}")
            z_y_relation_constraints.append(constr2)

    #z_y relationship constraints; start, end -- new model
    constr2_start = model.addConstr(path_duration[-2] >= 0,name=f"relation_{-2}")
    z_y_relation_constraints.append(constr2_start)

    constr2_end = model.addConstr(path_duration[-1] >= 0 + gp.quicksum(edge_cumu[j,-1] for j in incoming_adj_list[-1]),name=f"relation_{-1}")
    z_y_relation_constraints.append(constr2_end)



    cover_constraints = []
    upper_bound_path_duration = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            #Constraints - Node cover exactly once
            constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1,name=f"cover_{i}")
            cover_constraints.append(constr)


            #Upper bound on path duration - new model
            constr2 = model.addConstr(path_duration[i] <= max_duty_duration)
            upper_bound_path_duration.append(constr2)

    upper_bound_path_duration.append(path_duration[-1] <= max_duty_duration)

    #constraint on z -- new model 
    linearisation = defaultdict(list)
    for (i,j) in graph.edges():
        constr1 = model.addConstr(edge_cumu[i,j] <= max_duty_duration *edge_vars[i,j])
        constr2 = model.addConstr(edge_cumu[i,j] <= path_duration[i])
        # constr3 = model.addConstr(edge_cumu[i,j] >= 0) #implicit
        constr4 = model.addConstr(edge_cumu[i,j] >= path_duration[i] - (max_duty_duration* (1- edge_vars[i,j])))
        linearisation[i,j].append(constr1)
        linearisation[i,j].append(constr2)
        # linearisation[i,j].append(constr3)
        linearisation[i,j].append(constr4)

    
    if not show_logs:
        print("Number of decision variables: ", len(edge_vars))
        print("Number of flow constraints: ", len(flow_constraints))
        print("Number of cover constraints: ", len(cover_constraints))
        print("Number of linearisation constraints: ", len(linearisation)*3)
        print("Number of relationship constraints: ", len (z_y_relation_constraints))
        model.setParam('OutputFlag', 0)
    model.optimize()

    paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
    if show_roster_stats:
        roster_statistics(paths, service_dict)

    return paths, len(paths)

def mpc_duration_constr_lazy(graph, service_dict, show_logs = True, max_duty_duration=6*60, time_limit = 60, show_duties = False, show_roster_stats = False):

    model = gp.Model("MPC")
    if time_limit:
        model.setParam('TimeLimit', time_limit)

    incoming_var = defaultdict(list)
    outgoing_var = defaultdict(list)
    incoming_relation_var = defaultdict(list) 
    edge_vars = {} #xij - binary
    edge_cumu = {} #zij -continuous

    incoming_adj_list = nx.to_dict_of_lists(graph.reverse())
    # outgoing_adj_list = nx.to_dict_of_lists(graph)

    #Decision Variables
    for (i,j) in graph.edges():
        # print(i,j)
        edge_vars[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        incoming_var[j].append(edge_vars[i,j])
        outgoing_var[i].append(edge_vars[i,j])
        edge_cumu[i,j] = model.addVar(vtype=GRB.CONTINUOUS, lb = 0, name=f"z_{i}_{j}")
        incoming_relation_var[j].append(edge_cumu[i,j])

    path_duration = {}    
    for node in graph.nodes():
        path_duration[node] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{node}")

    # path_duration[-1] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{-1}")
    # path_duration[-2] = model.addVar(vtype=GRB.CONTINUOUS, name=f"y_{-2}")

    #Objective 
    model.setObjective(gp.quicksum(edge_vars[i,-1] for i in graph.nodes if i not in [-1, -2]), GRB.MINIMIZE)
    # print(incoming_var)

    #Constraints - Flow conservation
    flow_constraints = []
    z_y_relation_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(incoming_var[i])== 1,name=f"Service_flow_{i}") #gp.quicksum(outgoing_var[i]
            flow_constraints.append(constr)

            #z_y relationship constraints -- new model
            constr2 = model.addConstr(path_duration[i] >= service_dict[i].serv_dur + gp.quicksum(edge_cumu[j,i] for j in incoming_adj_list[i]),name=f"relation_{i}")
            # constr.Lazy = 2
            z_y_relation_constraints.append(constr2)

    #z_y relationship constraints; start, end -- new model
    constr2_start = model.addConstr(path_duration[-2] >= 0,name=f"relation_{-2}")
    # constr2_start.Lazy = 2
    z_y_relation_constraints.append(constr2_start)

    constr2_end = model.addConstr(path_duration[-1] >= 0 + gp.quicksum(edge_cumu[j,-1] for j in incoming_adj_list[-1]),name=f"relation_{-1}")
    # constr2_end.Lazy = 2
    z_y_relation_constraints.append(constr2_end)



    cover_constraints = []
    upper_bound_path_duration = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            #Constraints - Node cover exactly once
            constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1,name=f"cover_{i}")
            cover_constraints.append(constr)


            #Upper bound on path duration - new model
            constr2 = model.addConstr(path_duration[i] <= max_duty_duration)
            upper_bound_path_duration.append(constr2)

    upper_bound_path_duration.append(path_duration[-1] <= max_duty_duration)

    #constraint on z -- new model 
    linearisation = defaultdict(list)
    for (i,j) in graph.edges():
        constr1 = model.addConstr(edge_cumu[i,j] <= max_duty_duration *edge_vars[i,j])
        constr1.Lazy = 2
        constr2 = model.addConstr(edge_cumu[i,j] <= path_duration[i])
        constr2.Lazy = 2
        # constr3 = model.addConstr(edge_cumu[i,j] >= 0) #implicit
        constr4 = model.addConstr(edge_cumu[i,j] >= path_duration[i] - (max_duty_duration* (1- edge_vars[i,j])))
        constr4.Lazy = 2
        linearisation[i,j].append(constr1)
        linearisation[i,j].append(constr2)
        # linearisation[i,j].append(constr3)
        linearisation[i,j].append(constr4)

    
    if not show_logs:
        print("Number of decision variables: ", len(edge_vars))
        print("Number of flow constraints: ", len(flow_constraints))
        print("Number of cover constraints: ", len(cover_constraints))
        print("Number of linearisation constraints: ", len(linearisation)*3)
        print("Number of relationship constraints: ", len (z_y_relation_constraints))
        model.setParam('OutputFlag', 0)
    model.optimize()

    paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
    if show_roster_stats:
        roster_statistics(paths, service_dict)

    return paths, len(paths)


def lazy(graph, service_dict, show_logs = True, max_duty_duration=6*60, lazy_iterations =100, show_lazy_updates_every = 10, show_duties = False, show_roster_stats = False):
    model = gp.Model("Lazy")

    incoming_var = defaultdict(list)
    outgoing_var = defaultdict(list) 
    edge_vars = {} #xij - binary

    incoming_adj_list = nx.to_dict_of_lists(graph.reverse())

    #Decision Variables
    for (i,j) in graph.edges():
        edge_vars[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        incoming_var[j].append(edge_vars[i,j])
        outgoing_var[i].append(edge_vars[i,j])


    #Objective 
    model.setObjective(gp.quicksum(edge_vars[i,-1] for i in graph.nodes if i not in [-1, -2]), GRB.MINIMIZE)


    #Constraints - Flow conservation
    flow_constraints = []
    z_y_relation_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(incoming_var[i])== 1,name=f"Service_flow_{i}") #gp.quicksum(outgoing_var[i]
            flow_constraints.append(constr)



    cover_constraints = []
    for i in graph.nodes():
        if i in [-1,-2]: continue
        else:
            constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1,name=f"cover_{i}")
            cover_constraints.append(constr)

    if not show_logs:
        print("Number of decision variables: ", len(edge_vars))
        print("Number of flow constraints: ", len(flow_constraints))
        print("Number of cover constraints: ", len(cover_constraints))
        model.setParam('OutputFlag', 0)
    model.optimize()

    print("\n\nInitial Solve Completed!")
    paths, paths_decision_vars = generate_paths(outgoing_var, show_paths = False)
    roster_statistics(paths, service_dict)


    k=1
    lazy_constrs = [] 
    for i in range(lazy_iterations):

        bad_paths, bad_paths_decision_vars = get_bad_paths(paths, paths_decision_vars, service_dict)
        lazy_constraints = get_lazy_constraints(bad_paths, bad_paths_decision_vars, service_dict)

        ###Resolving the model
        if model.Status == GRB.OPTIMAL:

            for lazy_vars in lazy_constraints:
                # print(lazy_vars)
                constr = model.addConstr(gp.quicksum(lazy_vars)<= len(lazy_vars)-1,name=f"lazy_{k}")
                lazy_constrs.append(constr)
                k+=1
            model.reset()
            model.optimize()
            solve_time = model.Runtime

        paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)
        if i in [j for j in range(0,lazy_iterations, show_lazy_updates_every)]:
            print("\nLazy Constraints addition iteration number: ", i)
            print(f"Model solved in {solve_time} seconds.")
            print("Total number of constraints: ", len(model.getConstrs()))
            roster_statistics(paths, service_dict)

    if show_duties:
        paths, paths_decision_vars = generate_paths(outgoing_var, show_duties)

    if show_roster_stats:
        roster_statistics(paths, service_dict)

    return paths, len(paths)

#single method column generator
def column_generation(graph, service_dict, init_column_generator = "random", mpc_timeout =30, pricing_method = "bellman ford", other_columns = False,iterations = 100, verbose = False):

    if init_column_generator == "random":
        init_duties = []
        init_duties_1 = generate_initial_feasible_duties_random_from_services(service_dict.values(), num_duties=934, show_duties= False)
    elif init_column_generator == "mpc":
        init_duties_1 = generate_initial_feasible_duties_random_from_services(service_dict.values(), num_duties=934, show_duties= False)
        init_duties, duty_count = mpc_duration_constr(graph, service_dict, time_limit = mpc_timeout, show_logs = verbose, show_duties = False, show_roster_stats = False)
        mpc_sol = duty_count
    current_duties = init_duties + init_duties_1
    if verbose: 
        print("Current Duties: ", len(current_duties))



    for i in range(iterations):
        if verbose: 
            print("\nIteration: ", i)
        obj, duals, basis, selected_duties, selected_duties_vars = restricted_linear_program(service_dict, current_duties, show_solutions= False, show_objective = verbose)
        # duty, reduced_cost = generate_new_column(graph, service_dict, duals, method = pricing_method, verbose = verbose)
        
        start_time_1 = time.time()
        duty, reduced_cost = generate_new_column_2(graph, service_dict, duals, method = pricing_method, verbose = verbose)
        end_time_1 = time.time()

        if other_columns:
            start_time_2 = time.time()
            duty2, reduced_cost2 = generate_new_column_2(graph, service_dict, duals, method = "topological sort", verbose = verbose)
            end_time_2 = time.time()

        if duty in current_duties:
            print("Column already in current duties")
            break
        else:
            if verbose:
                print("Unique Column found!")
                print(f"Column Generated through main method in:  {end_time_1 - start_time_1:.6f} seconds")
                print("Generated duty Main:", duty, "Reduced cost Main (shortest path):", reduced_cost)
                duty_dur =0 
                for serv in duty:
                    duty_dur += service_dict[serv].serv_dur
                print("Duty Duration: ", mins2hhmm(duty_dur))
                
            if pricing_method == "bellman ford" and reduced_cost >= 0:
                print("Optimal solution found!")
                break

            if pricing_method == "topological sort" and reduced_cost <= 1:
                print("Cannot find any more columns with negative reduced cost using topological sort!")
                break
            if pricing_method == "dp" and reduced_cost <= 1:
                print("Optimal solution found!")
                break
            
            current_duties.append(duty)
            if verbose:
                print("Current Duties: ", len(current_duties))
        

        if other_columns and duty2 == duty:
            print("Other Column is same as main column! ")
        elif other_columns and duty2 != duty and duty2:
            if verbose:
                print("Unique Column found through other methods!")
                print(f"Column Generated through other methods in:  {end_time_2 - start_time_2:.6f} seconds")
                print("Generated duty Other:", duty2, "Reduced cost Other (shortest path):", reduced_cost2)
                duty_dur =0 
                for serv in duty2:
                    duty_dur += service_dict[serv].serv_dur
                print("Duty Duration: ", mins2hhmm(duty_dur))
                
            if reduced_cost2 <= 1:
                print("Cannot find any more columns with negative reduced cost using topological sort!")
                break

            
            current_duties.append(duty2)
            if verbose:
                print("Current Duties: ", len(current_duties))

    if verbose:
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    decimal_duties = 0 
    for var_name, var_value in selected_duties:
        # print(var_name, var_value)
        if var_value <1:
            decimal_duties += 1
        # print( variable.x)

    print("Selected Duty count: ",len(selected_duties))
    print("Decimal Duty count: ",decimal_duties)

    final_duties = {}
    for var_name, var_value in selected_duties:
        # final_duties.append(current_duties[int(var_name[1:])])
        final_duties[var_name] = current_duties[int(var_name[1:])]   #a dictionary of (var_name, duty (a list of service_nums))

    return mpc_sol, current_duties, final_duties, selected_duties, obj


#checks if new column generated is actually new
def column_generation2(graph, service_dict, init_column_generator = "random", mpc_timeout =30, pricing_method = "bellman ford", iterations = None, verbose = False):

    if init_column_generator == "random":
        init_duties = []
        init_duties_1 = generate_initial_feasible_duties_random_from_services(service_dict.values(), num_services=len(service_dict), show_duties= False)
        current_duties = init_duties + init_duties_1
    elif init_column_generator == "mpc":
        # init_duties_1 = generate_initial_feasible_duties_random_from_services(service_dict.values(), num_services=len(service_dict), show_duties= False)
        init_duties, duty_count = mpc_duration_constr(graph, service_dict, time_limit = mpc_timeout, show_logs = verbose, show_duties = False, show_roster_stats = False)
        mpc_sol = duty_count
        current_duties = init_duties
        print("Initial Duties: ", len(current_duties))

    
    if verbose: 
        print("Current Duties: ", len(current_duties))


    i=0
    while True:
        if verbose: 
            print("\nIteration: ", i)
        obj, duals, basis, selected_duties, selected_duties_vars = restricted_linear_program(service_dict, current_duties, show_solutions= False, show_objective = verbose)
        # duty, reduced_cost = generate_new_column(graph, service_dict, duals, method = pricing_method, verbose = verbose)
        
        start_time_1 = time.time()
        duty, reduced_cost = generate_new_column_2(graph, service_dict, duals, method = pricing_method, verbose = verbose)
        end_time_1 = time.time()


        if duty in current_duties:
            print("Column already in current duties")
            break
        else:
            if verbose:
                print("Unique Column found!")
                print(f"Column Generated through main method in:  {end_time_1 - start_time_1:.6f} seconds")
                print("Generated duty Main:", duty, "Reduced cost Main (shortest path):", reduced_cost)
                duty_dur =0 
                for serv in duty:
                    duty_dur += service_dict[serv].serv_dur
                print("Duty Duration: ", mins2hhmm(duty_dur))
                
            if pricing_method == "bellman ford" and reduced_cost >= 0:
                print("Optimal solution found!")
                break

            if pricing_method == "topological sort" and reduced_cost <= 1:
                print("Cannot find any more columns with negative reduced cost using topological sort!")
                start_time_dp = time.time()
                duty_1, reduced_cost_1 = generate_new_column_2(graph, service_dict, duals, method = "ip", verbose = verbose)
                end_time_dp = time.time()
                if reduced_cost_1 <=1:
                    print("Optimal solution found!")
                    break
                elif duty_1 not in current_duties:
                    if verbose:
                        print("Unique Column found!")
                        print(f"Column Generated through IP in:  {end_time_dp - start_time_dp:.6f} seconds")
                        print("Generated duty Main:", duty, "Reduced cost Main (shortest path):", reduced_cost)
                        duty_dur =0 
                        for serv in duty:
                            duty_dur += service_dict[serv].serv_dur
                        print("Duty Duration: ", mins2hhmm(duty_dur))
                    current_duties.append(duty_1)
                else:
                    print("Column from DP already in current duties")
                    break

            if pricing_method in ["dp", "ip"] and reduced_cost <= 1:
                print("Optimal solution found!")
                break
            
            current_duties.append(duty)
            if verbose:
                print("Current Duties: ", len(current_duties))
        if iterations and i == iterations:
            break
        else:
            i+=1
        

    if verbose:
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    decimal_duties = 0 
    for var_name, var_value in selected_duties:
        # print(var_name, var_value)
        if var_value <1:
            decimal_duties += 1
        # print( variable.x)

    print("Selected Duty count: ",len(selected_duties))
    print("Decimal Duty count: ",decimal_duties)

    final_duties = {}
    for var_name, var_value in selected_duties:
        # final_duties.append(current_duties[int(var_name[1:])])
        final_duties[var_name] = current_duties[int(var_name[1:])]   #a dictionary of (var_name, duty (a list of service_nums))

    return mpc_sol, current_duties, final_duties, selected_duties, obj


# uses IP if topological column not present - most relavant
def column_generation3(graph, service_dict, init_column_generator = "random", mpc_timeout =30, pricing_method = "bellman ford", iterations = None, verbose = False):

    if init_column_generator == "random":
        init_duties = []
        init_duties_1 = generate_initial_feasible_duties_random_from_services(service_dict.values(), num_duties=934, show_duties= False)
        current_duties = init_duties + init_duties_1
    elif init_column_generator == "mpc":
        init_duties_1 = generate_initial_feasible_duties_random_from_services(service_dict.values(), num_duties=934, show_duties= False)
        init_duties, duty_count = mpc_duration_constr(graph, service_dict, time_limit = mpc_timeout, show_logs = verbose, show_duties = False, show_roster_stats = False)
        mpc_sol = duty_count
        # old_obj = mpc_sol
        current_duties = init_duties + init_duties_1

    
    if verbose: 
        print("Current Duties: ", len(current_duties))


    i=0
    while True:
        if verbose: 
            print("\nIteration: ", i)
        obj, duals, basis, selected_duties, selected_duties_vars = restricted_linear_program(service_dict, current_duties, show_solutions= False, show_objective = verbose)
        # improvement = old_obj - obj
        # old_obj = obj
        # duty, reduced_cost = generate_new_column(graph, service_dict, duals, method = pricing_method, verbose = verbose)
        
        start_time_1 = time.time()
        duty, reduced_cost = generate_new_column_2(graph, service_dict, duals, method = pricing_method, verbose = verbose)
        end_time_1 = time.time()

        if verbose:
            if duty not in current_duties: print("Unique Column found!")
            else: break
            print(f"Column Generated through main method in:  {end_time_1 - start_time_1:.6f} seconds")
            print("Generated duty Main:", duty, "Reduced cost Main (shortest path):", reduced_cost)
            duty_dur =0 
            for serv in duty:
                duty_dur += service_dict[serv].serv_dur
            print("Duty Duration: ", mins2hhmm(duty_dur))
            
        if (pricing_method == "bellman ford" and reduced_cost >= 0):
            print("Optimal solution found!")
            break

        if (pricing_method == "topological sort" and reduced_cost <= 1):
            print("Cannot find any more columns with negative reduced cost using topological sort!")
            start_time_dp = time.time()
            duty_1, reduced_cost_1 = generate_new_column_2(graph, service_dict, duals, method = "ip", verbose = verbose)
            end_time_dp = time.time()
            if reduced_cost_1 <=1:
                print("Optimal solution found!")
                break
            elif duty_1 not in current_duties:
                if verbose:
                    print("Unique Column found!")
                    print(f"Column Generated through IP in:  {end_time_dp - start_time_dp:.6f} seconds")
                    print("Generated duty Main:", duty, "Reduced cost Main (shortest path):", reduced_cost)
                    duty_dur =0 
                    for serv in duty:
                        duty_dur += service_dict[serv].serv_dur
                    print("Duty Duration: ", mins2hhmm(duty_dur))
                current_duties.append(duty_1)
            else:
                print("Column from DP already in current duties")
                break

        if (pricing_method in ["dp", "ip"] and reduced_cost <= 1):
            print("Optimal solution found!")
            break
        
        current_duties.append(duty)
        if verbose:
            print("Current Duties: ", len(current_duties))
        if iterations and i == iterations:
            break
        else:
            i+=1
        

    if verbose:
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    decimal_duties = 0 
    for var_name, var_value in selected_duties:
        # print(var_name, var_value)
        if var_value <1:
            decimal_duties += 1
        # print( variable.x)

    print("Selected Duty count: ",len(selected_duties))
    print("Decimal Duty count: ",decimal_duties)

    final_duties = {}
    for var_name, var_value in selected_duties:
        # final_duties.append(current_duties[int(var_name[1:])])
        final_duties[var_name] = current_duties[int(var_name[1:])]   #a dictionary of (var_name, duty (a list of service_nums))

    # for var in selected_duties_vars:
    #     print(var.x)
    return mpc_sol, current_duties, final_duties, selected_duties, obj
    

# for heuristic - uses IP
def column_generation4(graph, service_dict, current_duties, selected_vars, pricing_method = "bellman ford", iterations = None, verbose = False):

    # if init_column_generator == "random":
    #     init_duties = []
    #     init_duties_1 = generate_initial_feasible_duties_random_from_services(service_dict.values(), num_services=len(service_dict), show_duties= False)
    #     current_duties = init_duties + init_duties_1
    # elif init_column_generator == "mpc":
    #     init_duties_1 = generate_initial_feasible_duties_random_from_services(service_dict.values(), num_services=len(service_dict), show_duties= False)
    #     init_duties, duty_count = mpc_duration_constr(graph, service_dict, time_limit = mpc_timeout, show_logs = verbose, show_duties = False, show_roster_stats = False)
    #     mpc_sol = duty_count
    #     current_duties = init_duties + init_duties_1
    #     print("Initial Duties: ", len(current_duties))

    
    # if verbose: 
    #     print("Current Duties: ", len(current_duties))


    i=0
    while True:
        if verbose: 
            print("\nIteration: ", i)
        # print(current_duties)
        obj, duals, basis, selected_duties, selected_duties_vars = restricted_linear_program_for_heuristic(service_dict, current_duties, selected_vars, show_solutions= False, show_objective = False)
        # duty, reduced_cost = generate_new_column(graph, service_dict, duals, method = pricing_method, verbose = verbose)
        
        start_time_1 = time.time()
        duty, reduced_cost = generate_new_column_2(graph, service_dict, duals, method = pricing_method, verbose = False)
        end_time_1 = time.time()


        if duty in current_duties:
            print("Column already in current duties")
            break
        else:
            if verbose:
                print("Unique Column found!")
                print(f"Column Generated through main method in:  {end_time_1 - start_time_1:.6f} seconds")
                print("Generated duty Main:", duty, "Reduced cost Main (shortest path):", reduced_cost)
                duty_dur =0 
                for serv in duty:
                    duty_dur += service_dict[serv].serv_dur
                print("Duty Duration: ", mins2hhmm(duty_dur))
                
            if pricing_method == "bellman ford" and reduced_cost >= 0:
                print("Optimal solution found!")
                break

            if pricing_method == "topological sort" and reduced_cost <= 1:
                print("Cannot find any more columns with negative reduced cost using topological sort!")
                start_time_dp = time.time()
                duty_1, reduced_cost_1 = generate_new_column_2(graph, service_dict, duals, method = "ip", verbose = verbose)
                end_time_dp = time.time()
                if reduced_cost_1 <=1:
                    print("Optimal solution found!")
                    break
                elif duty_1 not in current_duties:
                    if verbose:
                        print("Unique Column found!")
                        print(f"Column Generated through IP in:  {end_time_dp - start_time_dp:.6f} seconds")
                        print("Generated duty Main:", duty, "Reduced cost Main (shortest path):", reduced_cost)
                        duty_dur =0 
                        for serv in duty:
                            duty_dur += service_dict[serv].serv_dur
                        print("Duty Duration: ", mins2hhmm(duty_dur))
                    current_duties.append(duty_1)
                else:
                    print("Column from DP already in current duties")
                    break

            if pricing_method in ["dp", "ip"] and reduced_cost <= 1:
                print("Optimal solution found!")
                break
            
            current_duties.append(duty)
            if verbose:
                print("Current Duties: ", len(current_duties))
        if iterations and i == iterations:
            break
        else:
            i+=1
        

    if verbose:
        print ("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    decimal_duties = 0 
    for var_name, var_value in selected_duties:
        # print(var_name, var_value)
        if var_value <1:
            decimal_duties += 1
        # print( variable.x)

    print("Selected Duty count: ",len(selected_duties))
    print("Decimal Duty count: ",decimal_duties)

    final_duties = {}
    for var_name, var_value in selected_duties:
        # final_duties.append(current_duties[int(var_name[1:])])
        final_duties[var_name] = current_duties[int(var_name[1:])]   #a dictionary of (var_name, duty (a list of service_nums))

    return current_duties, final_duties, selected_duties, obj, basis

def cg_heuristics(graph, service_dict, current_duties, threshold, n = 3):
    selected_vars = []
    
    obj, duals, basis, selected_duties, selected_duties_vars = restricted_linear_program(service_dict, current_duties, show_solutions= False, show_objective = True)
    
    while len(selected_vars) < len(service_dict):

        if basis == {}:
            print("Basis is empty, no more variables to add")
            break
        added_vars = False

        for key, value in basis.items():
            if value > threshold:
                if key not in selected_vars:
                    print("key added in threshold method: ", key, value)
                    selected_vars += [key]
                    added_vars = True
        
        if not added_vars:      
            sorted_basis = dict(sorted(basis.items(), key=lambda item: item[1], reverse=True))
            print("sorted basis: ", sorted_basis)
            # selected_vars += [max((k for k in basis if basis[k] != 1), key=lambda k: basis[k])]
            top_n_basis = dict(list(sorted_basis.items())[:n])
            print(top_n_basis)
            for key, value in top_n_basis.items():
                if key not in selected_vars:
                    print("key added in sorted method: ", key, value)
                    selected_vars += [key]
                    added_vars = True
            # print(to_add)
            print("no more variables to add, choosing the max from basis")

        negative_duals = 0
        for key, value in duals.items():
            if value < 0:
                # print("ye raha negative dual: ", key, value, "\n")
                negative_duals += 1
        print("Number of negative duals: ", negative_duals)
        print("Number of selected variables: ", len(selected_vars))

        current_duties, final_duties, selected_duties, obj, basis = column_generation4(graph, service_dict, selected_vars=selected_vars, current_duties=current_duties, pricing_method = "topological sort", iterations = 1000, verbose = False)
        print("Current Objective: ", obj)

        print("\n=================================================================\n")
    return selected_vars, current_duties