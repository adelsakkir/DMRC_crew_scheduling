import csv
from datetime import timedelta
import networkx as nx
import random
import matplotlib.pyplot as plt
import inspect
import numpy as np
from collections import defaultdict
import random
import gurobipy as gp 
from gurobipy import GRB

class Service:
    def __init__(self, attrs):
        self.serv_num = int(attrs[0])
        self.train_num = attrs[1]
        self.start_stn = attrs[2]
        self.start_time = hhmm2mins(attrs[3])
        self.end_stn = attrs[4]
        self.end_time = hhmm2mins(attrs[5])
        self.direction = attrs[6]
        self.serv_dur = int(attrs[7])
        self.jurisdiction = attrs[8]
        self.stepback_train_num = attrs[9]
        self.serv_added = False
        self.break_dur = 0
        self.trip_dur = 0

def hhmm2mins(hhmm):
    ''' Convert time from HH:MM format to minutes '''
    h, m = map(int, hhmm.split(':'))
    return h*60 + m

def mins2hhmm(mins):
    ''' Convert time from minutes to HH:MM format '''
    h = mins // 60
    m = mins % 60
    return f"{h:02}:{m:02}"

def fetch_data(filename, partial=False, rakes=10):
    ''' Fetch data from the given CSV file '''
    services = []
    services_dict = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            serv_obj = Service(row)
            if partial:
                if serv_obj.train_num in [f"{700+i}" for i in range(rakes+1)]:
                    services.append(serv_obj)
                    services_dict[serv_obj.serv_num] = serv_obj
            else:
                services.append(serv_obj)
                services_dict[serv_obj.serv_num] = serv_obj
    return services, services_dict

def draw_graph_with_edges(graph, n=50):
    ''' Draw the first n edges of the given graph '''
    # Create a directed subgraph containing only the first n edges
    subgraph = nx.DiGraph()
    
    # Add the first n edges and associated nodes to the subgraph
    edge_count = 0
    for u, v in graph.edges():
        if edge_count >= n:
            break
        if u != -2 and v != -1:
            subgraph.add_edge(u, v)
            edge_count += 1

    # Plotting the directed subgraph
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(subgraph)  # Position nodes using the spring layout
    nx.draw_networkx_nodes(subgraph, pos, node_size=50, node_color='red')
    nx.draw_networkx_labels(subgraph, pos, font_size=15)
    nx.draw_networkx_edges(
        subgraph, pos, arrowstyle='->', arrowsize=20, edge_color='blue'
    )
    
    plt.title(f"First {n} Directed Edges of the Network")
    # plt.show()
    plt.savefig(f'first{n}edges.png')

def node_legal(service1, service2):
    ''' Check if two services can be connected '''

    if service1.stepback_train_num == "No Stepback":
        if service2.train_num == service1.train_num:
            if service1.end_stn == service2.start_stn and 0 <= (service2.start_time - service1.end_time) <= 15:
                return True
        else:
            if (service1.end_stn[:4] == service2.start_stn[:4]) and (service2.start_time >= service1.end_time) and (service2.start_time <= service1.end_time + 120):
                return True
        
    else:
        if service2.train_num == service1.stepback_train_num:
            if (service1.end_stn == service2.start_stn) and 0 <= (service2.start_time - service1.end_time) <= 15:
                return True
        else:
            if (service1.end_stn[:4] == service2.start_stn[:4]) and (service2.start_time >= service1.end_time) and (service2.start_time <= service1.end_time + 120):
                return True
    return False

def no_overlap(service1, service2):
    ''' Check if two services overlap in time '''
    return service1.end_time <= service2.start_time

def create_duty_graph(services):
    ''' 
        Creates a directed graph of services, with source and sink nodes at -2, -1 respectively

        Arguments: services - list of Service objects

        Returns: a directed graph G
    '''
    G = nx.DiGraph()

    for i, service1 in enumerate(services):
        G.add_node(service1.serv_num)

    G.add_node(-1) #end_node
    G.add_node(-2) #start_node

    for i, service1 in enumerate(services):
        for j, service2 in enumerate(services):
            if i != j:
                if node_legal(service1, service2):
                    G.add_edge(service1.serv_num, service2.serv_num, weight=service1.serv_dur)

    #end node edges
    for i, service in enumerate(services):
        G.add_edge(service.serv_num, -1, weight=service.serv_dur)

    #start node edges
    for i, service in enumerate(services):
        G.add_edge(-2, service.serv_num, weight=0)
        
    return G

def extract_nodes(var_name):

    parts = var_name.split('_')
    if len(parts) != 3 or parts[0] != 'x':
        raise ValueError(f"Invalid variable name format: {var_name}")
    
    start_node = int(parts[1])
    end_node = int(parts[2])
    
    return start_node, end_node

def generate_paths(outgoing_var, show_paths = False):

    paths = []
    paths_decision_vars = []
    current = -2
    for start_path in outgoing_var[-2]:
        current_path = []
        current_path_decision_vars = []
        if start_path.x !=1:continue
        else:
            start, end = extract_nodes(start_path.VarName)
            # current_path.append(start_path.VarName)
            current_path.append(end)
            current_path_decision_vars.append(start_path)
            # start, end = extract_nodes(start_path.VarName)
            current = end
            while current != -1:
                for neighbour_edge in outgoing_var[current]:
                    if neighbour_edge.x !=1:continue
                    else:
                        start, end = extract_nodes(neighbour_edge.VarName)
                        current_path.append(end)
                        # current_path.append(neighbour_edge.VarName)
                        current_path_decision_vars.append(neighbour_edge)
                        # start, end = extract_nodes(neighbour_edge.VarName)
                        current = end
            paths.append(current_path)
            current_path.pop()
            paths_decision_vars.append(current_path_decision_vars)
            if show_paths:
                print(current_path)
    return paths, paths_decision_vars

def solution_verify(services, duties):
    ''' 
    Checks if all services are assigned to a duty 
    '''
    flag = True
    for service in services:
        service_check = False
        for duty in duties:
            if service.serv_num in duty:
                service_check = True
                break
        if service_check == False:
            print(f"Service {service.serv_num} not assigned to any duty")
            flag= False
            break
    return flag

def roster_statistics(paths, service_dict):

    """
    service_dict: The dictionary of service times    
    """

    #1 Number of duties
    print("\nRoster Statistics:")
    print("Number of duties: ", len(paths))

    #2 Maximum number of services in a duty
    max_len_duty = 0
    min_len_duty = 1e9
    for duty in paths:
        if len(duty)>max_len_duty:
            max_len_duty = len(duty)
        if len(duty)<min_len_duty:
            min_len_duty = len(duty)

    print("Maximum number of services in a duty: ", max_len_duty-1)
    print("Minimum number of services in a duty: ", min_len_duty-1)

    #3 Maximum duration of a duty
    max_duration = 0
    min_duration = 1e9
    serv_dur_6 = 0
    serv_dur_8 = 0
    for duty in paths:
        current_duration = 0
        for service in duty:
            # start, end = extract_nodes(edge)
            if service != -2: current_duration += service_dict[service].serv_dur
        if current_duration > max_duration:
            max_duration = current_duration
        if current_duration < min_duration:
            min_duration = current_duration
        if current_duration > (6*60):
            serv_dur_6+=1
        if current_duration > (8*60):
            serv_dur_8+=1
            
    print("Maximum duration of duty: ", mins2hhmm(max_duration))
    print("Minimum duration of duty: ", mins2hhmm(min_duration))
    print("Duties with driving time more than 6hrs: ",  serv_dur_6)
    print("Duties with driving time more than 8hrs: ",  serv_dur_8)

def get_bad_paths(paths, paths_decision_vars, service_dict):
    bad_paths = []
    bad_paths_decision_vars = []
    for i in range(len(paths)):
        current_duration = 0
        for node in paths[i]:
            # start, end = extract_nodes(edge)
            # if node != -2: current_duration += service_dict[node].serv_dur
            if node != -2: current_duration += service_dict[node].serv_dur
        if current_duration > 6*60:
            bad_paths.append(paths[i])
            bad_paths_decision_vars.append(paths_decision_vars[i])

    return bad_paths, bad_paths_decision_vars

def get_lazy_constraints(bad_paths, bad_paths_decision_vars, service_dict):
    lazy_constraints = []
    for i in range(len(bad_paths)):
        current_duration = 0
        current_lazy_constr = []
        bad_paths[i].append(-2)  #to make the size of paths and path_decision_vars equal
        for j in range(len(bad_paths[i])):
            # start, end = extract_nodes(bad_paths[i][j])
            node = bad_paths[i][j]
            if node != -2: current_duration += service_dict[node].serv_dur
            current_lazy_constr.append(bad_paths_decision_vars[i][j])
            if current_duration > 6*60:
                lazy_constraints.append(current_lazy_constr)
                break

    return lazy_constraints

def can_append(duty, service):
    ''' Checking if service can be appended to duty or not '''
    # get the last service in the duty
    last_service = duty[-1]
    
    # check if the end station of the last service is the as the start station of the current service
    start_end_stn_tf = last_service.end_stn[:4] == service.start_stn[:4]
    
    # check if the start time of the current service is within 15 minutes of the end time of the last service
    start_end_time_tf = 0 <= (service.start_time - last_service.end_time) <= 15
    # check if the end station of the last service is the same as the start station of the current service after a break
    start_end_stn_tf_after_break = last_service.end_stn[:4] == service.start_stn[:4]
    # check if the start time of the current service is within 120 minutes of the end time of the last service, used for adding break logic
    start_end_time_within = 0 <= (service.start_time - last_service.end_time) <= 120

    # stepback train number check
    if last_service.stepback_train_num == "No StepBack":
        start_end_rake_tf = last_service.train_num == service.train_num
    else:
        start_end_rake_tf = last_service.stepback_train_num == service.train_num
    
    # checking for valid conditions and time limits
    if start_end_rake_tf and start_end_stn_tf and start_end_time_tf:    # if start end rake & station are same, and times are within limits
        time_dur = service.end_time - duty[0].start_time                # total duty duration
        cont_time_dur = sum([serv.serv_dur for serv in duty])           # continuous duty duration
        if cont_time_dur <= 180 and time_dur <= 445:                    # continuous duty <= 3 hrs, total duty <= 7 hrs 25 mins
            return True
    elif start_end_time_within and start_end_stn_tf_after_break:        # if start end station is same after a break, and times are within limits
        time_dur = service.end_time - duty[0].start_time                
        if time_dur <= 445:                                             # total duty <= 7 hrs 25 mins
            return True
    return False

def solve_RMLP(services, duties, threshold=0):
    '''
    Solves the RMLP 

    Arguments: services - list of Service objects,
            duties - list of duties
    
    Returns: selected_duties - list of selected duties,
            dual_values - list of dual values for each service,
            selected_duties_vars - list of selected duty variables
            objective_value = objective value of the iteration
    '''
    objective = 0
    model = gp.Model("CrewScheduling")
    model.setParam('OutputFlag', 0)
    
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=0, name=f"x{i}"))

    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    service_constraints = []
    for service_idx, service in enumerate(services):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) >= 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)

    model.optimize()

    # Step 5: Check the solution and retrieve dual values and selected duties
    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        # print("Optimal solution found")
        
        # Get the dual variables for each service constraint
        # dual_values = [constr.Pi for constr in service_constraints] 
        dual_values = {f"Service_{service.serv_num}": constr.Pi for service, constr in zip(services, service_constraints)}

        selected_duties_vars = [v.varName for v in model.getVars() if v.x > threshold]
        selected_duties = [v for v in model.getVars() if v.x > threshold]
        
        return selected_duties, dual_values, selected_duties_vars, objective.getValue()
    else:
        print("No optimal solution found.")
        return None, None, None, None

def new_duty_with_bellman_ford(graph, dual_values):

    '''
    Finds a new duty using NetworkX Bellman-Ford algorithm

    Arguments: graph - directed graph of services,
            dual_values - list of dual values for each service

    Returns: path - list of services in the new duty,
            length - length of the new duty,
            graph_copy - copy of the graph with adjusted edge weights
    '''
    graph_copy = graph.copy()
    for u, v in graph_copy.edges():
        if u != -2:
            service_idx_u = u
            # dual_u = dual_values[service_idx_u]
            dual_u = dual_values[f"Service_{service_idx_u}"]

            graph_copy[u][v]['weight'] = -(dual_u)  # Adjust edge weight by dual value
        # else:
        #     service_idx_u = -2
        #     dual_u = dual_values[service_idx_u]
        #     graph_copy[u][v]['weight'] = -(dual_u)
    

    path = nx.bellman_ford_path(graph_copy, -2, -1, weight='weight')
    length = nx.bellman_ford_path_length(graph_copy, -2, -1, weight='weight')

    return path, length, graph_copy

def count_overlaps(selected_duties, services):
    '''
    Checks the number of overlaps of services in selected_duties, and prints them

    Arguments: selected_duties - duties that are selected after column generation
               services - all services

    Returns: Boolean - False, if number of services != all services covered in selected_duties; else True
    '''
    services_covered = {}

    for service in services:
        services_covered[service.serv_num] = 0

    for duty in selected_duties:
        for service in duty:
            services_covered[service] += 1

    num_overlaps = 0
    num_services = 0
    for service in services_covered:
        if services_covered[service] > 1:
            num_overlaps += 1
        if services_covered[service] != 0:
            num_services += 1

    print(f"Number of duties selected: {len(selected_duties)}")
    print(f"Total number of services: {len(services)}")
    print(f"Number of services that overlap in duties: {num_overlaps}")
    print(f"Number of services covered in duties: {num_services}")

    if len(services) != num_services:
        return False
    else:
        return True
    
def solve_MIP(services, duties, threshold=0, cutoff= 100, mipgap = 0.01, timelimit = 600):
    '''
    Solves the RMLP 

    Arguments: services - list of Service objects,
            duties - list of duties
    
    Returns: selected_duties - list of selected duties,
            dual_values - list of dual values for each service,
            selected_duties_vars - list of selected duty variables
            objective_value = objective value of the iteration
    '''
    objective = 0
    model = gp.Model("CrewScheduling_IP")
    model.setParam('OutputFlag', 0)
    
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.BINARY, name=f"x{i}"))

    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    service_constraints = []
    for service_idx, service in enumerate(services):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) == 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)

    model.setParam('MIPGap', mipgap)
    model.setParam('TimeLimit', timelimit)
    model.setParam('MIPFocus', 1)
    model.setParam('Cutoff', cutoff)
    model.optimize()

    # Step 5: Check the solution and retrieve dual values and selected duties
    if model.status == GRB.OPTIMAL or model.status == gp.GRB.TIME_LIMIT:
        selected_duties = [i for i, var in enumerate(duty_vars) if var.x > 0.5]
        return model.ObjVal, selected_duties, model
    else:
        return None, None, model

class DynamicBundleStabilisation:
    def __init__(self, services, alpha = 0.5, max_bundle_size = 10):
        self.services = services
        self.service_indices = {service.serv_num: i for i, service in enumerate(services)}
        self.alpha = alpha
        self.max_bundle_size = max_bundle_size

        # initialising bundle
        self.bundle = []        # list of [dual values, objective values] pairs
        self.stability_center = None
        self.best_objective = float('inf')

        # initialise duals to 0
        self.current_duals = {f"Service_{service.serv_num}": 0 for service in services}

        # proximity parameter for trust region
        self.mu = 1.0

    def get_stabilised_duals(self, duals, objective_value):
        """
        Update the bundle and caluclate stabilised dual values
        Arguments:
            duals: dictionary of dual values
            objective_value: objective value of the current iteration
        Returns:
            stabilised_duals: dictionary of stabilised dual values
        """

        # first call
        if self.stability_center is None:
            self.stability_center = dict(duals)
            self.best_objective = objective_value
            return duals
        
        self.bundle.append((duals, objective_value))
        if len(self.bundle) > self.max_bundle_size:
            # remove the oldest element
            self.bundle.pop(0)

        # update the stability center if obj imporves significantly
        if objective_value < self.best_objective * 0.50:    # 50% improvement
            self.stability_center = dict(duals)
            self.best_objective = objective_value

            # reduce mu to promote exploration
            self.mu = max(0.2 * self.mu, 0.01)
        else:
            # increase for more stability
            self.mu = min(2 * self.mu, 10)      

        stabilised_duals = self.solve_bundle_subproblem()
        return stabilised_duals
    
    def solve_bundle_subproblem(self):
        """
        Solves the bundle subproblem to get stabilised duals
        Returns:
            stabilised_duals: dictionary of stabilised dual values
        """

        if not self.bundle:
            return self.stability_center
    
        model = gp.Model("BundleSubproblem")
        model.setParam('OutputFlag', 0)

        dual_vars = {}
        for service in self.services:
            dual_key = f"Service_{service.serv_num}"
            dual_vars[dual_key] = model.addVar(vtype = GRB.CONTINUOUS, lb = 0.0, name=dual_key)
        
        # auxiliary variable for bundle model
        v = model.addVar(vtype = GRB.CONTINUOUS, lb = -GRB.INFINITY, name="v")

        # linearisation constraints for the bundle
        for i, (bundle_dual, bundle_obj) in enumerate(self.bundle):
            expr = bundle_obj
            for service in self.services:
                dual_key = f"Service_{service.serv_num}"
                if dual_key in bundle_dual:
                    expr += dual_vars[dual_key] - bundle_dual[dual_key]

            model.addConstr(v >= expr, name=f"cut_{i}")
        
        # objective: minimize v + (mu/2) * ||dual_vars - stability_center||^2
        stability_term = 0
        for service in self.services:
            dual_key = f"Service_{service.serv_num}"
            if dual_key in self.stability_center:
                diff = dual_vars[dual_key] - self.stability_center[dual_key]
                stability_term += diff * diff

        model.setObjective(v + (self.mu/2) * stability_term, GRB.MINIMIZE)

        model.optimize()

        if model.status == GRB.OPTIMAL:
            # return the stabilised duals
            stabilised_duals = {key: var.x for key, var in dual_vars.items()}
            return stabilised_duals
        else:
            # if not optimized, return the stability center
            return self.stability_center
        
def solve_RMLP_with_bundle(services, duties, bundle_stabiliser, threshold = 0):
    '''
    Solves the RMLP with bundle stabilisation

    Arguments: services - list of Service objects,
            duties - list of duties
            bundle_stabiliser - object of DynamicBundleStabilisation class
    
    Returns: selected_duties - list of selected duties,
            stabilised_dual_values - dict of stabilised dual values for each service,
            selected_duties_vars - list of selected duty variables
            objective_value = objective value of the iteration
    '''
    objective = 0
    model = gp.Model("CrewScheduling")
    model.setParam('OutputFlag', 0)
    
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=0, name=f"x{i}"))

    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    service_constraints = []
    for service_idx, service in enumerate(services):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) >= 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)

    model.optimize()

    # Step 5: Check the solution and retrieve dual values and selected duties
    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
        return None, None, None, None
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        
        # Get the dual variables for each service constraint
        dual_values = {f"Service_{service.serv_num}": constr.Pi for service, constr in zip(services, service_constraints)}

        stabilised_dual_values = bundle_stabiliser.get_stabilised_duals(dual_values, objective.getValue())

        selected_duties_vars = [v.varName for v in model.getVars() if v.x > threshold]
        selected_duties = [v for v in model.getVars() if v.x > threshold]
        
        return selected_duties, stabilised_dual_values, selected_duties_vars, objective.getValue()
    else:
        print("No optimal solution found.")
        return None, None, None, None
    
def better_pricing_problem_solver(graph, duals, prev_paths, epsilon=1e-6):
    '''
    Finds a new duty using NetworkX Bellman-Ford algorithm while preventing cycling

    Arguments: graph - directed graph of services,
            dual_values - list of dual values for each service
            prev_paths - list of previously selected paths
            epsilon - small value to perturb the edge weights and break ties

    Returns: path - list of services in the new duty
            cost - reduced cost of the new duty
            graph_copy - copy of the graph with adjusted edge weights
    '''

    if prev_paths is None:
        prev_paths = []

    graph_copy = graph.copy()

    import random
    random.seed()

    for u, v in graph_copy.edges():
        base_cost = 1.0
        if u != -2:
            dual_u = duals[f"Service_{u}"]

            # calculate reduced cost: base_cost - dual_u
            perturbation = random.uniform(0, epsilon)
            reduced_cost = -dual_u + perturbation
            graph_copy[u][v]['weight'] = reduced_cost

    try:
        path = nx.bellman_ford_path(graph_copy, -2, -1, weight='weight')
        reduced_cost = nx.bellman_ford_path_length(graph_copy, -2, -1, weight='weight')

        if path in prev_paths:
            print("prev path mai hai")
            for attempt in range(5):  # Try up to 5 times to find new path
                # Add larger perturbations to edges
                for u, v in graph_copy.edges():
                    perturbation = random.uniform(0, 0.01 * (attempt + 1))
                    graph_copy[u][v]['weight'] += perturbation
                
                try:
                    alt_path = nx.bellman_ford_path(graph_copy, -2, -1, weight='weight')
                    alt_cost = nx.bellman_ford_path_length(graph_copy, -2, -1, weight='weight')

                    if alt_path not in prev_paths and alt_cost < 0:
                        return alt_path[1:-1], alt_cost, graph_copy
                except:
                    continue
            
            # if no better alt path has been found, check if the previous path is still worth adding again
            if reduced_cost < -0.01:
                return path[1:-1], reduced_cost, graph_copy
            else:
                return None, 0, graph_copy
            
        if reduced_cost < 0:
            return path[1:-1], reduced_cost, graph_copy
        else:
            return None, reduced_cost, graph_copy
        
    except nx.NetworkXNoPath:
        return None, 0, graph_copy