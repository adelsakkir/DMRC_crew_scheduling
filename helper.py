import csv
from datetime import timedelta
import networkx as nx
import random
import matplotlib.pyplot as plt
import inspect
from collections import defaultdict
import gurobipy as gp 
from gurobipy import GRB
from collections import defaultdict
from collections import deque

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
    h, m = map(int, hhmm.split(':'))
    return h*60 + m

def mins2hhmm(mins):
    h = mins // 60
    m = mins % 60
    return f"{h:02}:{m:02}"

def fetch_data(filename):
    services = []
    services_dict = {}
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            serv_obj = Service(row)
            services.append(serv_obj)
            services_dict[serv_obj.serv_num] = serv_obj
    return services, services_dict

def fetch_data_by_rake(filename, partial=False, rakes=10):
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
    # Create a directed subgraph containing only the first n edges
    subgraph = nx.DiGraph()
    
    # Add the first n edges and associated nodes to the subgraph
    edge_count = 0
    for u, v in graph.edges():
        if edge_count >= n:
            break
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

## checking if two services can be connected
def node_legal(service1, service2):
    if service1.stepback_train_num == "No Stepback":
        if service2.train_num == service1.train_num:
            if service1.end_stn == service2.start_stn and 0 <= (service2.start_time - service1.end_time) <= 15:
                return True
        else:
            if (service1.end_stn[:4] == service2.start_stn[:4]) and (service2.start_time >= service1.end_time + 30) and (service2.start_time <= service1.end_time + 150):
                return True
        
    else:
        if service2.train_num == service1.stepback_train_num:
            if (service1.end_stn == service2.start_stn) and (service1.end_time == service2.start_time):
                return True
        else:
            if (service1.end_stn[:4] == service2.start_stn[:4]) and (service2.start_time >= service1.end_time + 30 ) and (service2.start_time <= service1.end_time + 150):
                return True
    return False

def no_overlap(service1, service2):
    return service1.end_time <= service2.start_time

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

def create_duty_graph(services):
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

def solution_verify(services, duties, verbose =True):
    flag = True
    for service in services:
        service_check = False
        for duty in duties:
            if service.serv_num in duty:
                service_check = True
                break
        if service_check == False:
            if verbose:
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



### Helpers for column generation
def can_append(duty, service):
    last_service = duty[-1]
    
    start_end_stn_tf = last_service.end_stn == service.start_stn
    # print(service.start_time, last_service.end_time)
    start_end_time_tf = 5 <= (service.start_time - last_service.end_time) <= 15
    start_end_stn_tf_after_break = last_service.end_stn[:4] == service.start_stn[:4]
    start_end_time_within = 50 <= (service.start_time - last_service.end_time) <= 150

    if last_service.stepback_train_num == "No StepBack":
        start_end_rake_tf = last_service.train_num == service.train_num
    else:
        start_end_rake_tf = last_service.stepback_train_num == service.train_num
    
    # Check for valid conditions and time limits
    if start_end_rake_tf and start_end_stn_tf and start_end_time_tf:
        time_dur = service.end_time - duty[0].start_time
        cont_time_dur = sum([serv.serv_dur for serv in duty])
        if cont_time_dur <= 180 and time_dur <= 445:
            return True
    elif start_end_time_within and start_end_stn_tf_after_break:
        time_dur = service.end_time - duty[0].start_time
        if time_dur <= 445:
            return True
    return False

def restricted_linear_program(service_dict, duties, show_solutions = False, show_objective = False, warm_start_solution=None, t=0):

    # objective = 0
    model = gp.Model("CrewScheduling")
    model.setParam('OutputFlag', 0)

    
    ###Decision Variables
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=0, name=f"x{i}"))

    big_penalty = 1e6


    ### Objective
    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    
    ### Constraints
    service_constraints = []
    for service_idx, service in enumerate(service_dict.values()):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty) >= 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)

    

    ### Warm Start from previous solution
    # if warm_start_solution:
    #     for i in warm_start_solution.keys():
    #         duty_vars[i].VBasis = gp.GRB.BASIC
    #         duty_vars[i].Start = warm_start_solution[i]

    # for v in model.getVars():
    #     if v.VBasis == gp.GRB.BASIC:
    #         # ct+=1
    #         # basis[int(v.VarNAME[1:])]= v.x
    #         print(f"Variable '{v.VarName}' is in the basis") 

    model.optimize()



    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
        return None, None, None, None
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        model.write("model.lp") 
        if show_solutions:
            print("Optimal solution found")
        
        # Get the dual variables for each service constraint
        # dual_values = [constr.Pi for constr in service_constraints]
        duals = dict([(constr.ConstrName, constr.Pi) for constr in service_constraints])

        
        solution = [v.x for v in model.getVars()]
        # print("Hi" ,len(solution))
        selected_duties = [(v.varName, v.x) for v in model.getVars() if v.x > 0]
        selected_duties_vars = [v for v in model.getVars() if v.x > 0]
        
        ct = 0 
        basis = {}
        for v in model.getVars():
            if v.VBasis == gp.GRB.BASIC:
                ct+=1
                basis[int(v.VarNAME[1:])]= v.x

                

        if show_solutions:
            print("Positive Duties, 0: ", len(selected_duties))
            for variable in selected_duties_vars:
                print(variable.varName, variable.x)
        if show_objective:    
            print(f"Objective Value: {objective.getValue()}")

        return objective.getValue(), duals, basis, selected_duties, selected_duties_vars
    else:
        print("No optimal solution found.")
        return None, None, None, None

def restricted_linear_program_for_heuristic(service_dict, duties, selected_vars, show_solutions = False, show_objective = False, warm_start_solution=None, t=0):

    # objective = 0
    model = gp.Model("CrewScheduling")
    model.setParam('OutputFlag', 0)

    
    ###Decision Variables
    duty_vars = []
    for i in range(len(duties)):
        if i in selected_vars:
            duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=1, lb=1, name=f"x{i}"))
        else:
            duty_vars.append(model.addVar(vtype=GRB.CONTINUOUS, ub=GRB.INFINITY, lb=0, name=f"x{i}"))

    ### Objective
    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    ### Constraints
    service_constraints = []
    for service_idx, service in enumerate(service_dict.values()):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty)>= 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)

    ### Warm Start from previous solution
    # if warm_start_solution:
    #     for i in warm_start_solution.keys():
    #         duty_vars[i].VBasis = gp.GRB.BASIC
    #         duty_vars[i].Start = warm_start_solution[i]

    # for v in model.getVars():
    #     if v.VBasis == gp.GRB.BASIC:
    #         # ct+=1
    #         # basis[int(v.VarNAME[1:])]= v.x
    #         print(f"Variable '{v.VarName}' is in the basis") 

    model.optimize()

    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
        return None, None, None, None
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        model.write("model.lp") 
        if show_solutions:
            print("Optimal solution found")
        
        # Get the dual variables for each service constraint
        # dual_values = [constr.Pi for constr in service_constraints]
        duals = dict([(constr.ConstrName, constr.Pi) for constr in service_constraints])

        solution = [v.x for v in model.getVars()]
        # print("Hi" ,len(solution))
        selected_duties = [(v.varName, v.x) for v in model.getVars() if v.x > 0]
        selected_duties_vars = [v for v in model.getVars() if v.x > 0]
        
        ct = 0 
        basis = {}
        for v in model.getVars():
            if v.VBasis == gp.GRB.BASIC:
                ct+=1
                basis[int(v.VarNAME[1:])]= v.x

        if show_solutions:
            print("Positive Duties, 0: ", len(selected_duties))
            for variable in selected_duties_vars:
                print(variable.varName, variable.x)
        if show_objective:    
            print(f"Objective Value: {objective.getValue()}")

        return objective.getValue(), duals, basis, selected_duties, selected_duties_vars
    else:
        print("No optimal solution found.")
        return None, None, None, None

def generate_initial_feasible_duties_random_from_services(services, num_services, show_duties = False):

    feasible_duties = []

    # initial set of duties should cover all services
    # not checking for breaks
    for service1 in services:
        duty = [service1]
        for service2 in services:
            if service1.serv_num != service2.serv_num:
                if can_append(duty, service2):
                    duty.append(service2)
        feasible_duties.append(duty)

    # random_duties = random.sample(feasible_duties, num_services)
    serv_num_duty = []

    # to get duty in terms of service numbers
    for duty in feasible_duties:
        tt = []
        for serv in duty:
            tt.append(serv.serv_num)
        serv_num_duty.append(tt)
    if show_duties:
        print(serv_num_duty)
    return serv_num_duty


def generate_new_column(graph, service_dict, dual_values, method = "topological sort", verbose = False):
    

    if method == "topological sort":

        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 0
            else:
                # graph[u][v]['weight'] = dual_values[u]
                graph[u][v]['weight'] = dual_values["Service_" + str(u)]

        topo_order = list(nx.topological_sort(graph))

        longest_dist = {node: float('-inf') for node in graph.nodes}
        longest_dist[-2] = 0  # Distance to source is 0
        predecessor = {node: None for node in graph.nodes} 

        # Relax edges in topological order
        for u in topo_order:
            for v in graph.successors(u):  
                weight = graph[u][v].get('weight')  
                if longest_dist[v] < longest_dist[u] + weight:
                    longest_dist[v] = longest_dist[u] + weight
                    predecessor[v] = u

        
        shortest_path = []
        curr = -1

        while curr is not None:  
            shortest_path.append(curr)
            curr = predecessor[curr]
        # shortest_path.pop()
        shortest_path.reverse()  
        # shortest_path.pop()

        if verbose:
            path_duals = []
            # path_duals.append(graph[-2][shortest_path[0]]['weight'])
            for i in range(len(shortest_path)-1):
                path_duals.append(graph[shortest_path[i]][shortest_path[i+1]]['weight'])
            # path_duals.append(graph[shortest_path[-1]][-1]['weight'])

            print("Path Duals: ",path_duals)

        return shortest_path[1:-1], longest_dist[-1]
    
    elif method == "bellman ford":
        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 1
            else:
                # graph[u][v]['weight'] = -dual_values[u]
                graph[u][v]['weight'] = -dual_values["Service_" + str(u)]

        shortest_path = nx.shortest_path(graph, source=-2, target=-1, weight='weight', method = 'bellman-ford')
        shortest_distance = nx.shortest_path_length(graph, source=-2, target=-1, weight='weight', method = 'bellman-ford')

        if verbose:
            path_duals = []
            for i in range(len(shortest_path)-1):
                path_duals.append(graph[shortest_path[i]][shortest_path[i+1]]['weight'])

            print("Path Duals: ", path_duals)

        return shortest_path[1:-1], shortest_distance
                        


    else:
        raise NotImplementedError(f"Method {method} not implemented")
    
def generate_new_column_2(graph, service_dict, dual_values, method = "topological sort", verbose = False, time_constr = 6*60):

    

    if method == "topological sort":

        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 0
            else:
                # graph[u][v]['weight'] = dual_values[u]
                graph[u][v]['weight'] = dual_values["Service_" + str(u)]

        topo_order = list(nx.topological_sort(graph))

        longest_dist = {node: float('-inf') for node in graph.nodes}
        longest_dist[-2] = 0  # Distance to source is 0
        duration = {node: 0 for node in graph.nodes}
        predecessor = {node: None for node in graph.nodes} 

        # Relax edges in topological order
        for u in topo_order:
            for v in graph.successors(u):  
                weight = graph[u][v].get('weight')
                if v!=-1:
                    succ_dur = service_dict[v].serv_dur
                else:
                    succ_dur = 0  
                if longest_dist[v] < longest_dist[u] + weight and duration[u] + succ_dur <= 6*60:
                    longest_dist[v] = longest_dist[u] + weight
                    duration[v] = duration[u] + succ_dur
                    predecessor[v] = u

        
        shortest_path = []
        curr = -1

        while curr is not None:  
            shortest_path.append(curr)
            curr = predecessor[curr]
        # shortest_path.pop()
        shortest_path.reverse()  
        # shortest_path.pop()

        if verbose:
            path_duals = []
            # path_duals.append(graph[-2][shortest_path[0]]['weight'])
            for i in range(len(shortest_path)-1):
                path_duals.append(graph[shortest_path[i]][shortest_path[i+1]]['weight'])
            # path_duals.append(graph[shortest_path[-1]][-1]['weight'])

            print("Path Duals: ",path_duals)

        return shortest_path[1:-1], longest_dist[-1]
    
    elif method == "bellman ford":
        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 1
            else:
                # graph[u][v]['weight'] = -dual_values[u]
                graph[u][v]['weight'] = -dual_values["Service_" + str(u)]

        shortest_path = nx.shortest_path(graph, source=-2, target=-1, weight='weight', method = 'bellman-ford')
        shortest_distance = nx.shortest_path_length(graph, source=-2, target=-1, weight='weight', method = 'bellman-ford')

        if verbose:
            path_duals = []
            for i in range(len(shortest_path)-1):
                path_duals.append(graph[shortest_path[i]][shortest_path[i+1]]['weight'])

            print("Path Duals: ", path_duals)

        return shortest_path[1:-1], shortest_distance
                        

    elif method == "dp":

        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 0
            else:
                # graph[u][v]['weight'] = dual_values[u]
                graph[u][v]['weight'] = dual_values["Service_" + str(u)]

        dp_dict = defaultdict(dict)

        topo_order = list(nx.topological_sort(graph))
        # print(topo_order)
        # for time in range(time_constr):
        #     dp_dict[-2][time] = (0, None)

        for node in topo_order:
            for time in range(time_constr+1):
                if node == -2:
                    dp_dict[node][time] = (0, None)
                else:
                    best_pred = None
                    best = 0
                    for pred in graph.predecessors(node):
                        # if graph.nodes[pred]["service_time"] > time:
                        if pred in [-1,-2] :pred_dur = 0 
                        else: pred_dur = service_dict[pred].serv_dur
                        if pred_dur > time:
                            continue
                        # time_range = max(0, time - graph.nodes[node]["service_time"])
                        # if node in [-1,-2] :node_dur = 0 
                        # else: node_dur = service_dict[node].serv_dur
                        time_range = max(0, time - pred_dur)
                        for t in range(time_range + 1):
                            current = dp_dict[pred][t][0] + graph[pred][node]['weight']
                            # print("Pred: ", pred, ", Time: ", time, ", T: ",t , "Current: ",current, "dp_dict", dp_dict[pred][t])
                            if current >= best:
                                best = current
                                best_pred = pred
                    dp_dict[node][time] = (best, best_pred)
        # print(dp_dict[-1])


        #extracting path
        remaining_time = time_constr
        current = -1
        spprc = dp_dict[current][remaining_time][0]

        path = deque()

        while current != -2:
            path.appendleft(current)
            # print(current, remaining_time)
            pred = dp_dict[current][remaining_time][1]
            # remaining_time -= graph.nodes[current]["service_time"]
            if pred in [-1,-2] :pred_dur = 0 
            else: pred_dur = service_dict[pred].serv_dur
            
            remaining_time -= pred_dur
            # remaining_time = max(0, remaining_time)
            current =pred
            # print("Shortest Path: ", path)  

        print("Shortest Path Length: ", spprc)
        print("Shortest Path: ", path)

        return list(path)[:-1], spprc

    elif method == "ip":
        service_dict[-1] = Service([-1,-1, "x", "00:00", "x", "00:00","x",0, "x", "x"])
        service_dict[-2] = Service([-2,-2, "x", "00:00", "x", "00:00","x",0, "x", "x"])

        for u, v in graph.edges():
            if u == -2:
                graph[u][v]['weight'] = 0
            else:
                # graph[u][v]['weight'] = dual_values[u]
                graph[u][v]['weight'] = dual_values["Service_" + str(u)]

        model = gp.Model("SPPRC")
        model.setParam('OutputFlag', 0) 

        incoming_var = defaultdict(list)
        outgoing_var = defaultdict(list) 
        edge_vars = {} #xij - binary

        incoming_adj_list = nx.to_dict_of_lists(graph.reverse())

        #Decision Variables
        for i,j in graph.edges():
            edge_vars[i,j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
            incoming_var[j].append(edge_vars[i,j])
            outgoing_var[i].append(edge_vars[i,j])
            # print(edge_vars)
            # print("Edge: ", i, " ", j)
            # model.update()
            # print(edge_vars[(i,j)].VarName)

        #Objective 
        model.setObjective(gp.quicksum((edge_vars[i,j]*graph[i][j]['weight']) for i,j in graph.edges()), GRB.MAXIMIZE)


        #Constraints - Flow conservation
        flow_constraints = []
        for i in graph.nodes():
            if i == -2:
                constr = model.addConstr(gp.quicksum(outgoing_var[i]) == 1,name=f"Service_flow_{i}") 
                flow_constraints.append(constr)
            elif i == -1:
                constr = model.addConstr(gp.quicksum(incoming_var[i]) == 1,name=f"Service_flow_{i}") 
                flow_constraints.append(constr)
            else:
                constr = model.addConstr(gp.quicksum(incoming_var[i])== gp.quicksum(outgoing_var[i]),name=f"Service_inflow_{i}") 
                # constr2 = model.addConstr(gp.quicksum(outgoing_var[i]) ==1, name=f"Service_outflow_{i}")

                flow_constraints.append(constr)
                # flow_constraints.append(constr2)


        
        #resource constraint 
        # duration = sum([(edge_vars[i,j].x *service_dict[i].serv_dur) for i,j in graph.edges() if i not in [-1,-2]])
        # print("Total Duration: ", duration)
        model.addConstr(gp.quicksum((edge_vars[i,j]*service_dict[i].serv_dur) for i,j in graph.edges() if i !=-1) <= time_constr)
        

        # model.update()
        model.write("model_dp.lp")
        model.optimize()

        if model.status == GRB.INFEASIBLE:
            print ("Hehe!")
            print("Model is not feasible")
        #extracting path

        reduced_cost = 0
        path = []
        current = -2
        path.append(current) 
        while current != -1:
            for var in outgoing_var[current]:
                if var.x ==1:
                    _ , next = extract_nodes(var.VarName)
                    reduced_cost += graph[current][next]['weight']
                    path.append(next)
                    current = next
                    break

        del service_dict[-1]
        del service_dict[-2]

        return path[1:-1], reduced_cost
        
    else:
        raise NotImplementedError(f"Method {method} not implemented")
    
# def generate_new_column_3(graph, service_dict, dual_values, method = "dp", verbose = False, time_constr = 6*60):



def mip(service_dict, duties, show_solutions = True, show_objective = True,warm =140):


    model = gp.Model("CrewScheduling")
    # model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 600)

    
    ###Decision Variables
    duty_vars = []
    for i in range(len(duties)):
        duty_vars.append(model.addVar(vtype=GRB.BINARY, name=f"x{i}"))

    big_penalty = 1e6


    ### Objective
    model.setObjective(gp.quicksum(duty_vars), GRB.MINIMIZE)

    
    ### Constraints
    service_constraints = []
    for service_idx, service in enumerate(service_dict.values()):
        constr = model.addConstr(
            gp.quicksum(duty_vars[duty_idx] for duty_idx, duty in enumerate(duties) if service.serv_num in duty)>= 1,
            name=f"Service_{service.serv_num}")
        service_constraints.append(constr)


    ##warm start
    for i in range(warm):
        duty_vars[i].Start = 1


    model.optimize()



    if model.status == GRB.INFEASIBLE:
        print('Infeasible problem!')
        return None, None, None, None
    elif model.status == GRB.OPTIMAL:
        objective = model.getObjective()
        # model.write("model.lp") 
        if show_solutions:
            print("Optimal solution found")
        
        # Get the dual variables for each service constraint
        # dual_values = [constr.Pi for constr in service_constraints]
        # duals = dict([(constr.ConstrName, constr.Pi) for constr in service_constraints])

        
        solution = [v.x for v in model.getVars()]
        # print("Hi" ,len(solution))
        selected_duties = [(v.varName, v.x) for v in model.getVars() if v.x > 0]
        selected_duties_vars = [v for v in model.getVars() if v.x > 0]

        if show_solutions:
            print("Positive Duties, 0: ", len(selected_duties))
            for variable in selected_duties_vars:
                print(variable.varName, variable.x)
        if show_objective:    
            print(f"Objective Value: {objective.getValue()}")

        return objective.getValue(), selected_duties
    else:
        print("No optimal solution found.")
        return None, None