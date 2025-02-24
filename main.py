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
from helper import Service, hhmm2mins, mins2hhmm, fetch_data, draw_graph_with_edges, node_legal, no_overlap, create_duty_graph, extract_nodes, generate_paths, roster_statistics, solution_verify, restricted_linear_program, generate_initial_feasible_duties_random_from_services, generate_new_column

services, service_dict = fetch_data('./StepBackServices.csv')
graph = create_duty_graph(services)

repeat =True
while repeat:
    num = int(input("""\nEnter the Model you'd like to run (integer): 
                    
                    1. Simple MPC
                    2. MPC with Duration Constraint
                    3. Lazy
                    4. Column Generation
                    
                    Expecting input: """))
    if num == 1:
        print("\nSimple MPC Model")
        duties, duty_count = simple_mpc(graph, service_dict, show_logs = False, show_duties = False, show_roster_stats = True)
        # print(duties)
        # print(duty_count)
    elif num == 2:
        print("\nMPC with Duration Constraints")
        duties, duty_count = mpc_duration_constr(graph, service_dict, time_limit = 60, show_logs = True, show_duties = False, show_roster_stats = True)
        # print(duties)
        # print(duty_count)
    elif num == 3:
        print("\nLazy Model")
        duties, duty_count = lazy(graph, service_dict, show_logs = False, max_duty_duration=6*60, lazy_iterations =100, show_lazy_updates_every = 10, show_duties = False, show_roster_stats = True)
    elif num == 4:
        print("\nColumn Generation Model")
        selected_duties, obj = column_generation(graph, service_dict, init_column_generator = "random", pricing_method = "bellman ford", iterations = 10, verbose = True)
    else:
        print("\nInvalid Input")


    repeat = input("\nWould you like to run another model? (y/n): ").lower() == 'y'
