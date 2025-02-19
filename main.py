import csv
from datetime import timedelta
import gurobipy as gp 
from gurobipy import GRB
import networkx as nx
import random
import matplotlib.pyplot as plt
import inspect
from collections import defaultdict

from models import simple_mpc, mpc_duration_constr, lazy, column_generation
from helper import Service, hhmm2mins, mins2hhmm, fetch_data, draw_graph_with_edges, node_legal, no_overlap, create_duty_graph, extract_nodes, generate_paths, roster_statistics, solution_verify, solving_RMLP_art_vars_final

services, service_dict = fetch_data('./StepBackServices.csv')
graph = create_duty_graph(services)

    
###Simple MPC
# duties, duty_count = simple_mpc(graph, service_dict, show_logs = False, show_duties = False, show_roster_stats = True)

# print("\nNext Model")
# duties, duty_count = mpc_duration_constr(graph, service_dict, time_limit = 60, show_logs = True, show_duties = False, show_roster_stats = True)
# print(duties)
# print(solution_verify(services, duties))
print("\nNext Model")
duties, duty_count = lazy(graph, service_dict, show_logs = False, max_duty_duration=6*60, lazy_iterations =100, show_lazy_updates_every = 10, show_duties = False, show_roster_stats = False)

# column_generation()
# print(graph.nodes())


# selected_duties, dual_values, selected_duties_vars = solving_RMLP_art_vars_final(services, duties)
# print(min(dual_values), max(dual_values), dual_values.count(1.0), dual_values.count(0.0), len(selected_duties))
# distinct_dual_values = list(set(dual_values))
# print(distinct_dual_values)