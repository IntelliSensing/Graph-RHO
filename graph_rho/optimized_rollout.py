"""Optimized rolling-horizon inference for Graph-RHO.

This module extends the original L-RHO rollout with two paper-specific ideas:
1. a stronger CP-SAT backend for each subproblem; and
2. adaptive thresholding for deciding which overlap tasks to fix.
"""

import copy
import collections
import time

import numpy as np

from graph_rho.optimized_solver import adaptive_threshold, optimized_flexible_jss
from graph_rho.utils.lrho_paths import ensure_lrho_makespan_on_path

ensure_lrho_makespan_on_path()

from flexible_jss_main_common import sort_by_task_order, validate_jss_sol
from flexible_jss_data import get_rollout_data
from flexible_jss_data_common import get_rollout_prediction


def select_jss_data(sel_task_indices, jobs_data, jobs_solution=None):
    """Select subset of job data for the current subproblem."""
    if jobs_solution is None:
        jobs_solution = {}
    jobs_data_sel = {}
    num_fix = 0
    num_not_fix = 0

    for job_id, task_id in sel_task_indices:
        if job_id not in jobs_data_sel:
            jobs_data_sel[job_id] = {}

        if (job_id, task_id) in jobs_solution:
            alt_id = jobs_solution[(job_id, task_id)][0]
            jobs_data_alt = copy.deepcopy(jobs_data[job_id][task_id][alt_id])
            jobs_data_sel[job_id][task_id] = {alt_id: jobs_data_alt}
            num_fix += 1
        else:
            jobs_data_sel[job_id][task_id] = copy.deepcopy(jobs_data[job_id][task_id])
            num_not_fix += 1

    if len(jobs_solution) > 0:    
        print(f'RHO subproblem: {num_fix} fix, {num_not_fix} not fix')

    for job_id in jobs_data_sel:
        jobs_data_sel[job_id] = dict(sorted(jobs_data_sel[job_id].items(), key=lambda x: x[0]))
        
    return jobs_data_sel


def select_window(start_loc, num_tasks, sorted_task_indices, jobs_solution, window):
    """Select tasks for current window."""
    sel_task_indices = []
    sel_i_task_loc = []
    i_task = start_loc
    
    while len(sel_task_indices) < window and i_task < num_tasks:
        task_idx = sorted_task_indices[i_task]
        if task_idx not in jobs_solution:
            sel_task_indices.append(task_idx)
            sel_i_task_loc.append(i_task)
        i_task += 1

    return sel_task_indices, sel_i_task_loc


def exec_step(sorted_task_indices, sel_task_indices, sel_i_task_loc, num_tasks, 
              jobs_solution, jobs_solution_sel, machines_assignment, machines_assignment_sel, 
              machines_start_time, jobs_start_time, n_machines, step):
    """Execute step tasks from the current solution."""
    assigned_task_type = collections.namedtuple("assigned_task_type", "start job index duration")

    if len(jobs_solution) + len(sel_task_indices) >= num_tasks:
        step_task_indices = sel_task_indices
        start_loc = num_tasks
    else:
        step_task_indices = sel_task_indices[:step]
        sel_task_indices_with_start = [
            (jobs_solution_sel[(job_id, task_id)][2], job_id, task_id) 
            for job_id, task_id in sel_task_indices
        ]
        sel_task_indices_with_start.sort()
        step_task_indices = [(job_id, task_id) for _, job_id, task_id in sel_task_indices_with_start[:step]]
       
        start_loc = min(sel_i_task_loc)
        while start_loc < num_tasks and sorted_task_indices[start_loc] in step_task_indices:
            start_loc += 1

    for job_id, task_id in step_task_indices:
        selected_alt, machine, start_value, duration = jobs_solution_sel[(job_id, task_id)]
        jobs_solution[(job_id, task_id)] = copy.deepcopy(jobs_solution_sel[(job_id, task_id)])
        machines_assignment[machine].append(
            assigned_task_type(start=start_value, job=job_id, index=task_id, duration=duration)
        )
        machines_start_time[machine] = max(machines_start_time[machine], start_value + duration)
        jobs_start_time[job_id] = max(jobs_start_time[job_id], start_value + duration)
            
    return start_loc, jobs_solution, machines_assignment, machines_start_time, jobs_start_time


def get_model_prediction_with_adaptive_threshold(
    model, data, overlapping_task_indices, jobs_solution_sel,
    model_decode_strategy='argmax', model_th=0.5, model_topk_th=0.5,
    use_adaptive_threshold=False, adaptive_target_ratio=0.6
):
    """
    Get model predictions with optional adaptive threshold.
    
    Args:
        model: The prediction model
        data: Rollout data
        overlapping_task_indices: List of overlapping task tuples
        jobs_solution_sel: Current solution
        model_decode_strategy: 'argmax', 'topk', or 'sampling'
        model_th: Fixed threshold (used if not adaptive)
        model_topk_th: Top-k threshold fraction
        use_adaptive_threshold: If True, compute threshold adaptively
        adaptive_target_ratio: Target ratio of tasks to fix when adaptive
        
    Returns:
        set: Set of (job_id, task_id) tuples predicted to be fixed
    """
    import torch
    
    # Get raw predictions
    tasks_pred_fix = get_rollout_prediction(
        model, data, 
        decode_strategy=model_decode_strategy,
        threshold=model_th, 
        topk_th=model_topk_th
    )
    
    # If adaptive threshold is enabled, recompute with adaptive threshold
    if use_adaptive_threshold and model_decode_strategy == 'argmax':
        # Get model output probabilities
        from flexible_jss_data_common import FlexibleJSSDataset, get_dataloader
        
        dataset = FlexibleJSSDataset([data])
        loader = get_dataloader(dataset, batch_size=1, shuffle=False)
        
        model.eval()
        with torch.no_grad():
            for batch_data in loader:
                device = next(model.parameters()).device
                batch_data = batch_data.to(device)
                output = model(batch_data)
                probs = torch.sigmoid(output).squeeze()
                
                # Compute adaptive threshold
                adaptive_th = adaptive_threshold(
                    probs, 
                    target_fix_ratio=adaptive_target_ratio,
                    min_th=0.3,
                    max_th=0.9
                )
                
                print(f'[Adaptive] Using threshold {adaptive_th:.3f} (target ratio: {adaptive_target_ratio})')
                
                # Re-decode with adaptive threshold
                predictions = (probs >= adaptive_th).cpu().numpy()
                task_label_idx = batch_data.task_label_idx.cpu().numpy()
                tasks_tuple = data['tasks_tuple']
                
                tasks_pred_fix = set([
                    tasks_tuple[task_idx] 
                    for pred, task_idx in zip(predictions.flatten(), task_label_idx) 
                    if pred > 0
                ])
                break
    
    return tasks_pred_fix


def optimized_rolling_horizon(
    n_machines, n_jobs, jobs_data, 
    window, step, time_limit, stop_search_time,
    oracle_time_limit=30, oracle_stop_search_time=3,
    action_type='default', do_warm_start=False,
    first_frac=0.5, random_p=0.8, num_oracle_trials=1,
    train_data_dir='train_data_dir/train_data', data_idx=1,
    model=None, model_decode_strategy='argmax', 
    model_th=0.5, model_topk_th=0.5,
    include_model_time=False, n_cpus=1,
    # New optimization parameters
    use_solver_optimization=True,
    use_adaptive_threshold=False,
    adaptive_target_ratio=0.6,
    verbose=False,
    print_str=''
):
    """
    Optimized rolling horizon algorithm with CP-SAT improvements.
    
    New parameters:
        use_solver_optimization: If True, use optimized CP-SAT parameters
        use_adaptive_threshold: If True, use adaptive threshold for model predictions
        adaptive_target_ratio: Target ratio of tasks to fix when using adaptive threshold
        verbose: If True, print detailed optimization progress
        
    Returns:
        tuple: (jobs_solution, machines_assignment, solve_time, average_solve_time, make_span)
    """
    oracle_time_limit, oracle_stop_search_time = time_limit, stop_search_time
    
    if action_type == 'default':
        do_warm_start = False
    if 'warm_start' in action_type:
        do_warm_start = True
    
    sorted_task_indices = sort_by_task_order(jobs_data)

    # Tracking solutions
    jobs_solution = {}
    machines_assignment = collections.defaultdict(list)
    jobs_solution_sel = {}
    machines_assignment_sel = {}

    # Tracking stats
    solve_time, model_time, num_solves, num_tasks = 0, 0, 0, len(sorted_task_indices)
    
    machines_start_time = [0 for _ in range(n_machines)]
    jobs_start_time = [0 for _ in range(n_jobs)]
    prev_sel_task_indices = []
    start_loc = 0

    training_data = []

    if verbose:
        opt_status = "ON" if use_solver_optimization else "OFF"
        adapt_status = "ON" if use_adaptive_threshold else "OFF"
        print(f'[Opt] Solver optimization: {opt_status}, Adaptive threshold: {adapt_status}')
    
    print(f'*************** window {window} step {step} ***************')

    while len(jobs_solution) < num_tasks:
        sel_task_indices, sel_i_task_loc = select_window(
            start_loc, num_tasks, sorted_task_indices, jobs_solution, window
        )
        jobs_data_sel = select_jss_data(sel_task_indices, jobs_data)
        
        if len(jobs_solution) > 0 and action_type != 'default':
            overlapping_task_indices = [task for task in prev_sel_task_indices if task in sel_task_indices]
            print(f'# Overlapping tasks = {len(overlapping_task_indices)}, out of {len(sel_task_indices)}')
            
            overlapping_jobs_solution_all = {
                (job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                for job_id, task_id in overlapping_task_indices
            }
            overlapping_machines_assignment_all = {
                machine: [task for task in machines_assignment_sel[machine]
                         if (task.job, task.index) in overlapping_task_indices]
                for machine in machines_assignment_sel
            }
            
            if action_type == 'fix_all':
                overlapping_jobs_solution = overlapping_jobs_solution_all
                
            elif action_type == 'model':
                model_start_time = time.time()
                data = get_rollout_data(
                    jobs_data_sel, n_machines, machines_start_time,
                    jobs_start_time, overlapping_task_indices, 
                    overlapping_jobs_solution_all,
                    overlapping_machines_assignment_all
                )
                
                # Use optimized prediction with adaptive threshold
                tasks_pred_fix = get_model_prediction_with_adaptive_threshold(
                    model, data, overlapping_task_indices, jobs_solution_sel,
                    model_decode_strategy=model_decode_strategy,
                    model_th=model_th,
                    model_topk_th=model_topk_th,
                    use_adaptive_threshold=use_adaptive_threshold,
                    adaptive_target_ratio=adaptive_target_ratio
                )
                
                model_time += time.time() - model_start_time
                print(f'{len(tasks_pred_fix)} tasks predicted to be fixed out of {len(overlapping_task_indices)}')
                
                overlapping_jobs_solution = {
                    (job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                    for job_id, task_id in overlapping_task_indices
                    if (job_id, task_id) in tasks_pred_fix
                }
                
            elif action_type == 'first':
                num_first = int(first_frac * len(overlapping_task_indices))
                overlapping_jobs_solution = {
                    (job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                    for job_id, task_id in overlapping_task_indices[:num_first]
                }
                
            elif action_type == 'random':
                overlapping_jobs_solution = {
                    (job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                    for job_id, task_id in overlapping_task_indices
                    if random_p > 0 and np.random.rand() < random_p
                }
                
            else:
                overlapping_jobs_solution = {}
                
            print(f'Number of overlapping solutions {len(overlapping_jobs_solution)} out of {len(overlapping_task_indices)}')
            
            if not do_warm_start:
                jobs_data_sel = select_jss_data(sel_task_indices, jobs_data_sel, jobs_solution=overlapping_jobs_solution)
        else:
            overlapping_jobs_solution = {}

        # Use optimized solver
        jobs_solution_sel, machines_assignment_sel, solve_time_sel, objective_sel = optimized_flexible_jss(
            jobs_data_sel, n_machines, 
            time_limit=time_limit, 
            stop_search_time=stop_search_time,
            machines_start_time=machines_start_time, 
            jobs_start_time=jobs_start_time,
            do_warm_start=do_warm_start, 
            jobs_solution_warm_start=overlapping_jobs_solution,
            use_optimization=use_solver_optimization,
            verbose=verbose
        )

        if objective_sel == float('inf'):
            print('Infeasible!!')

        start_loc, jobs_solution, machines_assignment, machines_start_time, jobs_start_time = \
            exec_step(sorted_task_indices, sel_task_indices, sel_i_task_loc, num_tasks, 
                      jobs_solution, jobs_solution_sel, machines_assignment, machines_assignment_sel,
                      machines_start_time, jobs_start_time, n_machines, step)
        
        prev_sel_task_indices = copy.deepcopy(sel_task_indices)
        solve_time += solve_time_sel
        num_solves += 1

        print(f'*************** exec_tasks {len(jobs_solution)} total num tasks {num_tasks} '
              f'window range [{sel_i_task_loc[0]}, {sel_i_task_loc[-1]+1}] ***************\n')
  
    if include_model_time:
        solve_time += model_time

    average_solve_time = solve_time / num_solves
    make_span = max([
        jobs_solution[(job_id, task_id)][2] + jobs_solution[(job_id, task_id)][3] 
        for job_id, task_id in sorted_task_indices
    ])
       
    # Validate solution
    if not validate_jss_sol(jobs_data, n_machines, jobs_solution, machines_assignment, make_span, 
                            machines_start_time=None, jobs_start_time=None, check_final=False):
        print('Final solution is invalid!')
    else:
        print('Final solution is valid!')

    p_str = print_str if print_str != '' else f'{action_type.upper()}{" Warm Start " if do_warm_start else " "}'
    opt_str = "[OPT]" if use_solver_optimization else ""
    
    print(f'\n********************************* Rolling Horizon {p_str} {opt_str} (makespan {make_span}) *********************************')
    print(f'total solve time {solve_time:.2f} average solve time {average_solve_time:.2f}, number of solves {num_solves}, makespan {make_span}')
    
    if 'model' in action_type: 
        print(f'model time {model_time:.2f} ({"Include" if include_model_time else "Exclude"})')
    print('\n')

    return jobs_solution, machines_assignment, solve_time, average_solve_time, make_span

