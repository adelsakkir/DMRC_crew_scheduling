import csv
from datetime import timedelta
import gurobipy as gp 
from gurobipy import GRB
import networkx as nx
import random
import matplotlib.pyplot as plt
import inspect
from collections import defaultdict
import argparse

from models import simple_mpc, mpc_duration_constr, mpc_duration_constr_lazy, lazy, column_generation, column_generation2, column_generation3
from helper import Service, hhmm2mins, mins2hhmm, fetch_data, draw_graph_with_edges, node_legal, no_overlap, create_duty_graph, extract_nodes, generate_paths, roster_statistics, solution_verify, restricted_linear_program, generate_initial_feasible_duties_random_from_services, generate_new_column



# Set up argument parser
parser = argparse.ArgumentParser(description="Run crew scheduling models.")
# parser.add_argument("model", type=int, choices=[1, 2, 3, 4], help="Choose a model: 1 (Simple MPC), 2 (MPC Duration), 3 (Lazy), 4 (Column Generation)")

parser.add_argument("--mpc_time", type=int, default=300, help="Time limit for MPC")
parser.add_argument("--col_iter", type=int, default=1000, help="Number of iterations for Column Generation (default: 10000)")

args = parser.parse_args()

# num = args.model
mpc_time  = args.mpc_time
col_iter = args.col_iter


services, service_dict = fetch_data('./StepBackServices.csv')
graph = create_duty_graph(services)

repeat =True
while repeat:
    num = int(input("""\nEnter the Model you'd like to run (integer): 
                    
                    1. Simple MPC
                    2. MPC with Duration Constraint
                    3. MPC Lazy
                    4. Lazy
                    5. Column Generation
                    
                    Expecting input: """))
    if num == 1:
        print("\nSimple MPC Model")
        duties, duty_count = simple_mpc(graph, service_dict, show_logs = False, show_duties = False, show_roster_stats = True)
        # print(duties)
        # print(duty_count)
    elif num == 2:
        print("\nMPC with Duration Constraints")
        duties, duty_count = mpc_duration_constr(graph, service_dict, time_limit = 112*60, show_logs = True, show_duties = False, show_roster_stats = True)
        # print(duties)
        # print(duty_count)
    elif num ==3:
        print("\nMPC with Duration Constraints + Lazy")
        duties, duty_count = mpc_duration_constr_lazy(graph, service_dict, time_limit = 112*60, show_logs = True, show_duties = False, show_roster_stats = True)
    elif num == 4:
        print("\nLazy Model")
        duties, duty_count = lazy(graph, service_dict, show_logs = False, max_duty_duration=6*60, lazy_iterations =100, show_lazy_updates_every = 10, show_duties = False, show_roster_stats = True)
    elif num == 5:
        print("\nColumn Generation Model")
        # duties, selected_duties, obj = column_generation(graph, service_dict, init_column_generator = "random", pricing_method = "bellman ford", iterations = 10, verbose = True)
        # mpc_sol, column_pool, duties, selected_duties, obj = column_generation(graph, service_dict, init_column_generator = "mpc", mpc_timeout = mpc_time,pricing_method = "topological sort", iterations = col_iter, verbose = True)
        # mpc_sol, column_pool, duties, selected_duties, obj = column_generation(graph, service_dict, init_column_generator = "mpc", mpc_timeout = mpc_time,pricing_method = "dp", iterations = col_iter, verbose = True)
        # mpc_sol, column_pool, duties, selected_duties, obj = column_generation(graph, service_dict, init_column_generator = "mpc", mpc_timeout = mpc_time,pricing_method = "topological sort", verbose = True)
        mpc_sol, column_pool, duties, selected_duties, obj = column_generation3(graph, service_dict, init_column_generator = "mpc", mpc_timeout = mpc_time,pricing_method = "topological sort", verbose = True)
        roster_statistics(duties.values(), service_dict)
    else:
        print("\nInvalid Input")

    repeat =False

    # repeat = input("\nWould you like to run another model? (y/n): ").lower() == 'y'
