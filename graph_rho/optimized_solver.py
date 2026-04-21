"""Optimized CP-SAT backend used by Graph-RHO rollout evaluation.

The solver keeps the original L-RHO formulation but tunes CP-SAT parameters
for better makespan quality under rolling-horizon time limits.
"""

import os
import random
import time
import copy
import threading
import collections
import numpy as np
from collections import defaultdict
from ortools.sat.python import cp_model


EPS = 1e-5


class OptimizedSolutionCallback(cp_model.CpSolverSolutionCallback):
    """Enhanced solution callback with detailed tracking."""

    def __init__(self, verbose=False):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0
        self.__solution_time = []
        self.__solution_obj = []
        self.__last_improvement_time = float('inf')
        self.__best_objective = float('inf')
        self.__verbose = verbose

    def on_solution_callback(self):
        """Called at each new solution."""
        current_time = self.WallTime()
        current_obj = self.ObjectiveValue()
        
        self.__solution_count += 1
        self.__solution_time.append(current_time)
        self.__solution_obj.append(current_obj)

        # Check for improvement
        if current_obj < self.__best_objective:
            improvement = self.__best_objective - current_obj
            self.__best_objective = current_obj
            self.__last_improvement_time = time.time()
            
            if self.__verbose:
                print(f"  [Opt] Solution {self.__solution_count}: obj={current_obj}, "
                      f"improvement={improvement:.0f}, time={current_time:.2f}s")

    @property
    def last_improvement_time(self):
        return self.__last_improvement_time
    
    @property
    def best_objective(self):
        return self.__best_objective
    
    @property
    def solution_count(self):
        return self.__solution_count


def get_optimized_solver_params(time_limit, problem_size='medium', optimize_for_makespan=True):
    """
    Get optimized solver parameters based on problem characteristics.
    
    Args:
        time_limit: Maximum solving time in seconds
        problem_size: 'small' (<100 tasks), 'medium' (100-500), 'large' (>500)
        optimize_for_makespan: If True, prioritize solution quality over speed
        
    Returns:
        dict of solver parameters
    """
    params = {}
    
    # Dynamic worker count based on CPU
    num_cpus = os.cpu_count() or 4
    params['num_search_workers'] = min(num_cpus, 8)
    
    # Time limit
    params['max_time_in_seconds'] = time_limit
    
    # Linearization level (0-2, higher = stronger propagation)
    params['linearization_level'] = 2
    
    # Random seed for reproducibility
    params['random_seed'] = 42
    
    # Problem size specific tuning
    # cp_model_probing_level: 0=no probing, 1=light, 2=normal, 3=strong
    if problem_size == 'small':
        params['cp_model_probing_level'] = 2  # More probing for small problems
    elif problem_size == 'large':
        params['cp_model_probing_level'] = 1  # Less probing for large problems
    else:  # medium
        params['cp_model_probing_level'] = 2
    
    if optimize_for_makespan:
        # Use more workers for better solution search
        params['num_search_workers'] = min(num_cpus, 12)
        
    return params


def calculate_tight_horizon(jobs_data, machines_start_time=None, jobs_start_time=None):
    """
    Calculate a tighter horizon estimate for better constraint propagation.
    
    Returns:
        int: Estimated horizon value
    """
    # Method 1: Sum of minimum durations per job (lower bound on makespan)
    job_lengths = []
    for job_id, job in jobs_data.items():
        job_length = sum(min(alt[0] for alt in task.values()) for task in job.values())
        job_lengths.append(job_length)
    
    # Method 2: Machine load estimation
    machine_load = defaultdict(int)
    for job_id, job in jobs_data.items():
        for task_id, task in job.items():
            # Assume we pick the machine with minimum duration
            min_duration = min(alt[0] for alt in task.values())
            best_machine = min(task.values(), key=lambda x: x[0])[1]
            machine_load[best_machine] += min_duration
    
    max_machine_load = max(machine_load.values()) if machine_load else 0
    max_job_length = max(job_lengths) if job_lengths else 0
    
    # Use the maximum of both estimates with safety margin
    base_horizon = max(max_job_length, max_machine_load)
    
    # Add offset from start times
    offset = 0
    if jobs_start_time is not None:
        offset = max(offset, max(jobs_start_time) if isinstance(jobs_start_time, (list, tuple)) else max(jobs_start_time.values()))
    if machines_start_time is not None:
        offset = max(offset, max(machines_start_time) if isinstance(machines_start_time, (list, tuple)) else max(machines_start_time.values()))
    
    # Safety factor of 1.5 to ensure feasibility
    tight_horizon = int((base_horizon + offset) * 1.5)
    
    return tight_horizon


def optimized_flexible_jss(jobs_data, n_machines, time_limit=-1, stop_search_time=5,
                           machines_start_time=None, jobs_start_time=None,
                           do_warm_start=False, jobs_solution_warm_start=None,
                           use_optimization=True, verbose=False):
    """
    Optimized flexible job shop scheduling solver.
    
    Args:
        jobs_data: Job data dictionary
        n_machines: Number of machines
        time_limit: Maximum solving time
        stop_search_time: Stop if no improvement in this many seconds
        machines_start_time: Machine availability times
        jobs_start_time: Job release times
        do_warm_start: Whether to use warm start hints
        jobs_solution_warm_start: Previous solution for warm start
        use_optimization: If True, use optimized solver parameters
        verbose: If True, print detailed solving progress
        
    Returns:
        tuple: (jobs_solution, machines_assignment, solve_time, objective)
    """
    if jobs_solution_warm_start is None:
        jobs_solution_warm_start = {}

    model = cp_model.CpModel()

    # Calculate horizon
    if use_optimization:
        # Use tighter horizon for better propagation
        horizon = calculate_tight_horizon(jobs_data, machines_start_time, jobs_start_time)
        # Fallback to original method if tight horizon seems too small
        original_horizon = sum(
            max(alt[0] for alt in task.values())
            for job in jobs_data.values()
            for task in job.values()
        )
        offset = 0
        if jobs_start_time is not None:
            offset = max(jobs_start_time) if isinstance(jobs_start_time, (list, tuple)) else max(jobs_start_time.values())
        if machines_start_time is not None:
            m_offset = max(machines_start_time) if isinstance(machines_start_time, (list, tuple)) else max(machines_start_time.values())
            offset = max(offset, m_offset)
        original_horizon += offset
        horizon = max(horizon, original_horizon // 2)  # Use at least half of original
    else:
        horizon = 0
        for job_id, job in jobs_data.items():
            for task_id, task in job.items():
                max_task_duration = max(alt[0] for alt in task.values())
                horizon += max_task_duration
        offset = 0
        if jobs_start_time is not None:
            offset = max(jobs_start_time) if isinstance(jobs_start_time, (list, tuple)) else max(jobs_start_time.values())
        if machines_start_time is not None:
            m_offset = max(machines_start_time) if isinstance(machines_start_time, (list, tuple)) else max(machines_start_time.values())
            offset = max(offset, m_offset)
        horizon += offset

    if verbose:
        print(f"[Opt] Horizon = {horizon}")

    # Global storage of variables
    intervals_per_resources = collections.defaultdict(list)
    starts = {}
    ends = {}
    presences = {}
    job_ends = []
    
    # For optimized warm start
    local_starts = {}  # Store local start variables for warm start hints

    # Build model
    for job_id, job in jobs_data.items():
        previous_end = None
        if len(job) == 0:
            continue
            
        for task_id, task in job.items():
            min_duration = min(alt[0] for alt in task.values())
            max_duration = max(alt[0] for alt in task.values())

            suffix_name = "_j%i_t%i" % (job_id, task_id)
            
            st_task = 0
            if jobs_start_time is not None:
                if isinstance(jobs_start_time, (list, tuple)):
                    st_task = jobs_start_time[job_id]
                else:
                    st_task = jobs_start_time.get(job_id, 0)

            start = model.NewIntVar(st_task, st_task + horizon, "start" + suffix_name)
            duration = model.NewIntVar(min_duration, max_duration, "duration" + suffix_name)
            end = model.NewIntVar(st_task, st_task + horizon, "end" + suffix_name)

            starts[(job_id, task_id)] = start
            ends[(job_id, task_id)] = end

            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end

            l_presences = []
            for alt_id, alt in task.items():
                alt_suffix = "_j%i_t%i_a%i" % (job_id, task_id, alt_id)
                l_presence = model.NewBoolVar("presence" + alt_suffix)

                l_machine = alt[1]
                l_st_task = 0
                if machines_start_time is not None:
                    if isinstance(machines_start_time, (list, tuple)):
                        l_st_task = machines_start_time[l_machine]
                    else:
                        l_st_task = machines_start_time.get(l_machine, 0)
                        
                l_start = model.NewIntVar(l_st_task, l_st_task + horizon, "start" + alt_suffix)
                l_duration = alt[0]
                l_end = model.NewIntVar(l_st_task, l_st_task + horizon, "end" + alt_suffix)

                l_interval = model.NewOptionalIntervalVar(
                    l_start, l_duration, l_end, l_presence, "interval" + alt_suffix
                )

                l_presences.append(l_presence)

                model.Add(start == l_start).OnlyEnforceIf(l_presence)
                model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                model.Add(end == l_end).OnlyEnforceIf(l_presence)

                intervals_per_resources[l_machine].append(l_interval)
                presences[(job_id, task_id, alt_id)] = l_presence
                local_starts[(job_id, task_id, alt_id)] = l_start

                # Enhanced warm start: add hints for both presence AND start time
                if do_warm_start and (job_id, task_id) in jobs_solution_warm_start:
                    warm_sol = jobs_solution_warm_start[(job_id, task_id)]
                    warm_alt_id = warm_sol[0]
                    
                    if warm_alt_id == alt_id:
                        model.AddHint(l_presence, 1)
                        # Optimized: also hint start time for better convergence
                        if use_optimization and len(warm_sol) >= 3:
                            warm_start_time = warm_sol[2]
                            model.AddHint(l_start, warm_start_time)
                    else:
                        model.AddHint(l_presence, 0)

            model.AddExactlyOne(l_presences)
        
        job_ends.append(previous_end)

    # Machine no-overlap constraints
    for machine_id in range(n_machines):
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # Configure solver
    solver = cp_model.CpSolver()
    
    if use_optimization:
        # Get optimized parameters
        num_tasks = sum(len(job) for job in jobs_data.values())
        if num_tasks < 100:
            problem_size = 'small'
        elif num_tasks > 500:
            problem_size = 'large'
        else:
            problem_size = 'medium'
            
        opt_params = get_optimized_solver_params(time_limit, problem_size, optimize_for_makespan=True)
        
        for param_name, param_value in opt_params.items():
            setattr(solver.parameters, param_name, param_value)
            
        if verbose:
            print(f"[Opt] Using optimized solver with {opt_params['num_search_workers']} workers, "
                  f"linearization={opt_params['linearization_level']}, "
                  f"probing={opt_params['cp_model_probing_level']}")
    else:
        if time_limit > 0:
            solver.parameters.max_time_in_seconds = time_limit
            solver.parameters.num_search_workers = 4

    # Solve with early stopping
    if stop_search_time >= time_limit:
        status = solver.Solve(model)
        model.status = status
    else:
        def solve_model(model, solver, callback):
            status = solver.Solve(model, callback)
            model.status = status

        solution_callback = OptimizedSolutionCallback(verbose=verbose)
        solver_thread = threading.Thread(target=solve_model, args=(model, solver, solution_callback))
        solver_thread.start()

        while solver_thread.is_alive():
            time.sleep(0.1)
            if time.time() - solution_callback.last_improvement_time > stop_search_time:
                solution_callback.StopSearch()
                if verbose:
                    print(f"[Opt] No improvement in {stop_search_time}s, stopping. "
                          f"Best obj: {solution_callback.best_objective}")
                solver_thread.join()

    # Extract solution
    status = model.status
    jobs_solution = {}
    machines_assignment = collections.defaultdict(list)
    assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for job_id, job in jobs_data.items():
            for task_id, task in job.items():
                start_value = solver.Value(starts[(job_id, task_id)])
                machine, duration, selected_alt = -1, -1, -1
                
                for alt_id, alt in task.items():
                    if solver.Value(presences[(job_id, task_id, alt_id)]):
                        duration = alt[0]
                        machine = alt[1]
                        selected_alt = alt_id

                jobs_solution[(job_id, task_id)] = (selected_alt, machine, start_value, duration)
                machines_assignment[machine].append(
                    assigned_task_type(start=start_value, job=job_id, index=task_id, duration=duration)
                )

        for machine in machines_assignment:
            machines_assignment[machine].sort()
        objective = solver.ObjectiveValue()
    else:
        print("[Opt] No solution found.")
        objective = float('inf')

    solve_time = solver.WallTime()
    
    if verbose:
        print(f"[Opt] Status: {solver.StatusName(status)}")
        print(f"[Opt] Objective: {objective}")
        print(f"[Opt] Stats - conflicts: {solver.NumConflicts()}, "
              f"branches: {solver.NumBranches()}, time: {solve_time:.2f}s")

    return jobs_solution, machines_assignment, solve_time, objective


def adaptive_threshold(probs, target_fix_ratio=0.6, min_th=0.3, max_th=0.9):
    """
    Compute adaptive threshold based on probability distribution.
    
    This ensures we fix approximately target_fix_ratio of tasks,
    which can lead to better makespan by not fixing too many or too few.
    
    Args:
        probs: Tensor or array of predicted probabilities
        target_fix_ratio: Target fraction of tasks to fix (default 0.6)
        min_th: Minimum threshold (default 0.3)
        max_th: Maximum threshold (default 0.9)
        
    Returns:
        float: Computed threshold
    """
    import torch
    
    if len(probs) == 0:
        return 0.5
    
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy().flatten()
    else:
        probs_np = np.array(probs).flatten()
    
    # Sort probabilities in descending order
    sorted_probs = np.sort(probs_np)[::-1]
    
    # Find threshold that fixes target_fix_ratio of tasks
    k = int(len(sorted_probs) * target_fix_ratio)
    k = max(1, min(k, len(sorted_probs) - 1))
    
    threshold = sorted_probs[k]
    
    # Clip to reasonable range
    threshold = max(min_th, min(max_th, threshold))
    
    return float(threshold)

