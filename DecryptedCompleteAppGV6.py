global top_solutions  # inserted
pass
pass
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import pulp
import numpy as np
import os
import traceback
salary_cap = 50000
team_limit = 5
num_individuals = 3000
num_generations = 40
top_n_solutions = 100
mutation_rate = 0.075
crossover_probability = 1
top_solutions = []

class Individual:
    def __init__(self, position=None, captain=None):
        if position is None:
            self.position = np.random.choice(range(len(df)), 6, replace=False)
        else:  # inserted
            self.position = position
        if captain is None:
            self.captain = np.random.choice(self.position)
        else:  # inserted
            self.captain = captain
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        team_limit = 5
        selected_players = df.iloc[self.position]
        cptn = df.iloc[self.captain]
        total_salary = df.iloc[self.position]['SALARY'].sum() + df.iloc[self.captain]['SALARY'] * 0.5
        if total_salary > salary_upper or total_salary < salary_lower:
            return (-1)
        for team in df['TEAM'].unique():
            if sum(selected_players['TEAM'] == team) > team_limit:
                return (-1)
        else:  # inserted
            total_daily_matchup = selected_players['DAILY MATCHUP'].sum()
            if total_daily_matchup > matchup_upper or total_daily_matchup < matchup_lower:
                return (-1)
            total_consistency = selected_players['CONSISTENCY'].sum()
            if total_consistency > consistency_upper or total_consistency < consistency_lower:
                return (-1)
            total_proj_ownership = selected_players['PROJ. OWNERSHIP'].sum()
            if total_proj_ownership > ownership_upper or total_proj_ownership < ownership_lower:
                return (-1)
            total_proj_fps = selected_players['PROJ. FPS'].sum()
            if total_proj_fps > fps_upper or total_proj_fps < fps_lower:
                return (-1)
            total_proj_ceiling = selected_players['PROJ. CEILING'].sum()
            if total_proj_ceiling > ceiling_upper or total_proj_ceiling < ceiling_lower:
                return (-1)
            total_proj_floor = selected_players['PROJ. FLOOR'].sum()
            if total_proj_floor > floor_upper or total_proj_floor < floor_lower:
                return (-1)
            total_proj_fps_avg = selected_players['FPS AVG'].sum()
            if total_proj_fps_avg > fps_avg_upper or total_proj_fps_avg < fps_avg_lower:
                return (-1)
            total_opp_v_pos = selected_players['OPP V POS'].sum()
            if total_opp_v_pos > opp_v_pos_upper or total_opp_v_pos < opp_v_pos_lower:
                return (-1)
            total_safe = selected_players['SAFE'].sum()
            if total_safe > safe_upper or total_safe < safe_lower:
                return (-1)
            total_lev = selected_players['LEV'].sum()
            if total_lev > lev_upper or total_lev < lev_lower:
                return (-1)
            total_val = selected_players['VAL'].sum()
            if total_val > val_upper or total_val < val_lower:
                return (-1)
            total_fl_val = selected_players['FL. VAL'].sum()
            if total_fl_val > fl_val_upper or total_fl_val < fl_val_lower:
                return (-1)
            total_ceil_val = selected_players['CEIL. VAL'].sum()
            if total_ceil_val > ceil_val_upper or total_ceil_val < ceil_val_lower:
                return (-1)
            return selected_players['PROJ. FPS'].sum()

    def crossover(self, other, crossover_probability=0.8):
        if np.random.rand() > crossover_probability:
            if np.random.rand() > 0.5:
                return self
            return other
        crossover_point = np.random.randint(1, len(self.position) - 1)
        child_position = np.concatenate([self.position[:crossover_point], other.position[crossover_point:]])
        child_position = np.unique(child_position)
        if len(child_position) < 6:
            missing = np.random.choice(list(set(range(len(df))) - set(child_position)), 6 - len(child_position), replace=False)
            child_position = np.concatenate([child_position, missing])
        child = Individual()
        child.position = child_position
        child.captain = np.random.choice(child.position)
        child.fitness = child.evaluate_fitness()
        return child

    def mutate(self):
        for i in range(len(self.position)):
            if np.random.rand() < mutation_rate:
                swap_with = np.random.randint(0, len(self.position))
                self.position[i], self.position[swap_with] = (self.position[swap_with], self.position[i])
        self.captain = np.random.choice(self.position)
        self.fitness = self.evaluate_fitness()

def select_parents(population, fitness):
    min_fitness = np.min(fitness)
    if min_fitness < 0:
        fitness = fitness - min_fitness + 1
    fitness_sum = np.sum(fitness)
    if fitness_sum == 0:
        selection_probs = np.ones(len(fitness)) / len(fitness)
    else:  # inserted
        selection_probs = fitness / fitness_sum
    parents = np.random.choice(population, size=2, p=selection_probs, replace=False)
    return parents

def is_solution_unique(new_solution, top_solutions):
    new_solution_sorted = sorted(new_solution)
    for existing_solution, _, _ in top_solutions:
        if new_solution_sorted == sorted(existing_solution):
            return False
    else:  # inserted
        return True

def is_number(s):
    """Check if the string is a number."""  # inserted
    try:
        float(s)
        return True
    except ValueError:
        return False
    else:  # inserted
        pass

def is_valid_range(s):
    """Check if the input is a valid range like \'65 - 75\' or a single number."""  # inserted
    if '-' in s:
        parts = s.split('-')
        if len(parts) == 2 and is_number(parts[0].strip()) and is_number(parts[1].strip()):
            return True
    return is_number(s)

def validate_inputs():
    global ceiling_lower  # inserted
    global consistency_lower  # inserted
    global generate_lineups  # inserted
    global fl_val_upper  # inserted
    global fps_avg_lower  # inserted
    global ownership_upper  # inserted
    global safe_lower  # inserted
    global val_lower  # inserted
    global fps_lower  # inserted
    global ceil_val_lower  # inserted
    global fl_val_lower  # inserted
    global floor_upper  # inserted
    global val_upper  # inserted
    global lev_lower  # inserted
    global floor_lower  # inserted
    global ceil_val_upper  # inserted
    global opp_v_pos_lower  # inserted
    global salary_lower  # inserted
    global safe_upper  # inserted
    global ownership_lower  # inserted
    global lev_upper  # inserted
    global matchup_upper  # inserted
    global opp_v_pos_upper  # inserted
    global matchup_lower  # inserted
    global fps_avg_upper  # inserted
    global fps_upper  # inserted
    global salary_upper  # inserted
    global consistency_upper  # inserted
    global ceiling_upper  # inserted
    try:
        salary_input = salary_entry.get().replace(',', '')
        matchup_input = matchup_entry.get()
        consistency_input = consistency_entry.get()
        ownership_input = ownership_entry.get()
        fps_input = fps_entry.get()
        ceiling_input = ceiling_entry.get()
        floor_input = floor_entry.get()
        fps_avg_input = fps_avg_entry.get()
        opp_v_pos_input = opp_v_pos_entry.get()
        safe_input = safety_entry.get()
        lev_input = leverage_entry.get()
        val_input = value_entry.get()
        fl_val_input = floor_val_entry.get()
        ceil_val_input = ceiling_val_entry.get()
        gl_input = lineups_entry.get()
        salary_input = salary_input if salary_input else '25000-50000'
        matchup_input = matchup_input if matchup_input else '0-500'
        consistency_input = consistency_input if consistency_input else '0-500'
        ownership_input = ownership_input if ownership_input else '0-500'
        fps_input = fps_input if fps_input else '0-500'
        ceiling_input = ceiling_input if ceiling_input else '0-500'
        floor_input = floor_input if floor_input else '0-500'
        fps_avg_input = fps_avg_input if fps_avg_input else '0-500'
        opp_v_pos_input = opp_v_pos_input if opp_v_pos_input else '0-500'
        safe_input = safe_input if safe_input else '0-1000'
        lev_input = lev_input if lev_input else '0-500'
        val_input = val_input if val_input else '0-500'
        fl_val_input = fl_val_input if fl_val_input else '0-500'
        ceil_val_input = ceil_val_input if ceil_val_input else '0-500'
        gl_input = gl_input if gl_input else '10'
        generate_lineups = int(gl_input)
        if is_valid_range(salary_input):
            if not 25000 <= float(salary_input.split('-')[0].strip()) <= 50000:
                raise ValueError('Salary must be a valid number or range between $25,000 and $50,000')
            if is_valid_range(matchup_input):
                if not 0 <= float(matchup_input.split('-')[0].strip()) <= 500:
                    raise ValueError('Daily MatchUp must be a valid number or range between 0 and 500')
                if is_valid_range(consistency_input):
                    if not 0 <= float(consistency_input.split('-')[0].strip()) <= 500:
                        raise ValueError('Consistency must be a valid number or range between 0 and 500')
                    if is_valid_range(ownership_input):
                        if not 0 <= float(ownership_input.split('-')[0].strip()) <= 500:
                            raise ValueError('Projected Ownership must be a valid number or range between 0 and 500')
                        if not is_valid_range(fps_input):
                            raise ValueError('Projected FPS must be a valid number or range')
                        if '-' in fps_input:
                            fps_parts = [float(part.strip()) for part in fps_input.split('-')]
                            if 0 <= fps_parts[0] <= 500:
                                pass  # postinserted
                            else:  # inserted
                                raise ValueError('Each value in the Projected FPS range must be between 0 and 500')
                        else:  # inserted
                            fps_value = float(fps_input)
                            if not 0 <= fps_value <= 500:
                                raise ValueError('Projected FPS must be between 0 and 500')
                        if is_valid_range(ceiling_input):
                            if not 0 <= float(ceiling_input.split('-')[0].strip()) <= 500:
                                raise ValueError('Projected Ceiling must be a valid number or range between 0 and 500')
                            if is_valid_range(floor_input):
                                if not 0 <= float(floor_input.split('-')[0].strip()) <= 500:
                                    raise ValueError('Projected Floor must be a valid number or range between 0 and 500')
                                if is_valid_range(fps_avg_input):
                                    if not 0 <= float(fps_avg_input.split('-')[0].strip()) <= 500:
                                        raise ValueError('FPS AVG must be a valid number or range between 0 and 500')
                                    if is_valid_range(opp_v_pos_input):
                                        if not 0 <= float(opp_v_pos_input.split('-')[0].strip()) <= 500:
                                            raise ValueError('FPS AVG must be a valid number or range between 0 and 500')
                                        if is_valid_range(safe_input):
                                            if not 0 <= float(safe_input.split('-')[0].strip()) <= 1000:
                                                raise ValueError('FPS AVG must be a valid number or range between 0 and 500')
                                            if is_valid_range(lev_input):
                                                if not 0 <= float(lev_input.split('-')[0].strip()) <= 500:
                                                    raise ValueError('FPS AVG must be a valid number or range between 0 and 500')
                                                if is_valid_range(val_input):
                                                    if not 0 <= float(val_input.split('-')[0].strip()) <= 500:
                                                        raise ValueError('FPS AVG must be a valid number or range between 0 and 500')
                                                    if is_valid_range(fl_val_input):
                                                        if not 0 <= float(fl_val_input.split('-')[0].strip()) <= 500:
                                                            raise ValueError('FPS AVG must be a valid number or range between 0 and 500')
                                                        if int(generate_lineups) > 150:
                                                            raise ValueError('Generate Line Ups must be a valid number or range between 1 and 150')
                                                        messagebox.showinfo('Success', 'All inputs are valid!')
                                                        salary_lower, salary_upper = parse_range_input(salary_input)
                                                        matchup_lower, matchup_upper = parse_range_input(matchup_input)
                                                        consistency_lower, consistency_upper = parse_range_input(consistency_input)
                                                        ownership_lower, ownership_upper = parse_range_input(ownership_input)
                                                        fps_lower, fps_upper = parse_range_input(fps_input)
                                                        ceiling_lower, ceiling_upper = parse_range_input(ceiling_input)
                                                        floor_lower, floor_upper = parse_range_input(floor_input)
                                                        fps_avg_lower, fps_avg_upper = parse_range_input(fps_avg_input)
                                                        opp_v_pos_lower, opp_v_pos_upper = parse_range_input(opp_v_pos_input)
                                                        safe_lower, safe_upper = parse_range_input(safe_input)
                                                        lev_lower, lev_upper = parse_range_input(lev_input)
                                                        val_lower, val_upper = parse_range_input(val_input)
                                                        fl_val_lower, fl_val_upper= parse_range_input(fl_val_input)
                                                        ceil_val_lower, ceil_val_upper = parse_range_input(ceil_val_input)

    except ValueError as e:
        messagebox.showerror('Input Error', str(e))
    return

def parse_range_input(input_value):
    """Parse the input value into a lower and upper bound."""  # inserted
    if '-' in input_value:
        parts = input_value.split('-')
        lower_bound = float(parts[0].strip())
        upper_bound = float(parts[1].strip())
        return (lower_bound, upper_bound)
    lower_bound = upper_bound = float(input_value)
    return (lower_bound, upper_bound)

def upload_file():
    global df  # inserted
    file_path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv')], title='Choose a CSV file')
    if file_path:
        try:
            df = pd.read_csv(file_path)
            messagebox.showinfo('File Uploaded', 'CSV file successfully uploaded!')
            print('I am here ')
            columns_to_check = ['SALARY', 'DAILY MATCHUP', 'CONSISTENCY', 'PROJ. OWNERSHIP', 'PROJ. FPS', 'PROJ. CEILING', 'PROJ. FLOOR', 'FPS AVG', 'OPP V POS', 'SAFE', 'LEV', 'VAL', 'FL. VAL', 'CEIL. VAL']
            if 'SALARY' in df:
                print(df['SALARY'])
                df['SALARY'] = df['SALARY'].str.replace(',', '').astype(int)
                df['SALARY'] = pd.to_numeric(df['SALARY'])
            else:  # inserted
                messagebox.showerror('Error', 'SALARY columns is required in CSV')
            print('I am here')
            total_bool = 0
            for col in columns_to_check:
                if col in df:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    print(df[col])
                    if df[col].isnull().any():
                        messagebox.showerror('Error', f'Column \'{col}\' contains non-numeric values.')
                        total_bool = total_bool + 1
                    else:  # inserted
                        t = 1
                else:  # inserted
                    messagebox.showerror('Error', f'Column \'{col}\' does not exist in CSV.')
        except Exception as e:
            messagebox.showerror('Error', f'Failed to read CSV file: {str(e)}')
    else:  # inserted
        messagebox.showwarning('No File Selected', 'Please select a CSV file to upload.')

def solve_lp():
    prob = pulp.LpProblem('Lineup_Optimization', pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts('Player', df.index, cat='Binary')
    captain_var = pulp.LpVariable.dicts('Captain', df.index, cat='Binary')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'PROJ. FPS'] for i in df.index]), 'Total FPS')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'SALARY'] + captain_var[i] * df.loc[i, 'SALARY'] * 0.5 for i in df.index]) <= salary_upper, 'Salary Cap')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'SALARY'] + captain_var[i] * df.loc[i, 'SALARY'] * 0.5 for i in df.index]) >= salary_lower, 'Salary Lower')
    prob += (pulp.lpSum([player_vars[i] for i in df.index]) == 6, 'Select Exactly 6 Players')
    prob += (pulp.lpSum([captain_var[i] for i in df.index]) == 1, 'Select Exactly 1 Captain')
    for i in df.index:
        prob += (captain_var[i] <= player_vars[i], f'Captain Must Be Selected Player {i}')
    for team in df['TEAM'].unique():
        prob += (pulp.lpSum([player_vars[i] for i in df.index if df.loc[i, 'TEAM'] == team]) <= team_limit, f'Team Limit {team}')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'DAILY MATCHUP'] for i in df.index]) >= matchup_lower, 'Daily Matchup Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'DAILY MATCHUP'] for i in df.index]) <= matchup_upper, 'Daily Matchup Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'CONSISTENCY'] for i in df.index]) >= consistency_lower, 'Consistency Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'CONSISTENCY'] for i in df.index]) <= consistency_upper, 'Consistency Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'PROJ. OWNERSHIP'] for i in df.index]) >= ownership_lower, 'Ownership Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'PROJ. OWNERSHIP'] for i in df.index]) <= ownership_upper, 'Ownership Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'PROJ. FPS'] for i in df.index]) >= fps_lower, 'Projected FPS Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'PROJ. FPS'] for i in df.index]) <= fps_upper, 'Projected FPS Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'PROJ. CEILING'] for i in df.index]) >= ceiling_lower, 'Projected Ceiling Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'PROJ. CEILING'] for i in df.index]) <= ceiling_upper, 'Projected Ceiling Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'PROJ. FLOOR'] for i in df.index]) >= floor_lower, 'Projected Floor Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'PROJ. FLOOR'] for i in df.index]) <= floor_upper, 'Projected Floor Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'FPS AVG'] for i in df.index]) >= fps_avg_lower, 'FPS Avg Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'FPS AVG'] for i in df.index]) <= fps_avg_upper, 'FPS Avg Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'OPP V POS'] for i in df.index]) >= opp_v_pos_lower, 'OPP V POS Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'OPP V POS'] for i in df.index]) <= opp_v_pos_upper, 'OPP V POS Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'SAFE'] for i in df.index]) >= safe_lower, 'SAFE Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'SAFE'] for i in df.index]) <= safe_upper, 'SAFE Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'LEV'] for i in df.index]) >= lev_lower, 'LEV Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'LEV'] for i in df.index]) <= lev_upper, 'LEV Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'VAL'] for i in df.index]) >= val_lower, 'VAL Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'VAL'] for i in df.index]) <= val_upper, 'VAL Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'FL. VAL'] for i in df.index]) >= fl_val_lower, 'FL VAL Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'FL. VAL'] for i in df.index]) <= fl_val_upper, 'FL VAL Upper Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'CEIL. VAL'] for i in df.index]) >= ceil_val_lower, 'CEIL VAL Lower Bound')
    prob += (pulp.lpSum([player_vars[i] * df.loc[i, 'CEIL. VAL'] for i in df.index]) <= ceil_val_upper, 'CEIL VAL Upper Bound')
    prob.solve()
    solution_indices = [i for i in df.index if player_vars[i].varValue == 1]
    captain_index = [i for i in df.index if captain_var[i].varValue == 1]
    return (solution_indices, captain_index[0]) if len(solution_indices) == 6 and len(captain_index) == 1 else (solution_indices, captain_index[i])

def start_optimizer():
    global top_solutions  # inserted
    bool1 = validate_inputs()
    firstflag = 1
    if 'df' in globals():
        salary_value = salary_entry.get()
    else:  # inserted
        messagebox.showinfo('File Upload', 'CSV not uploaded')
    if len(df) < 7 or bool1 == 0 or len(df) > 38:
        messagebox.showinfo('File Upload', 'Data needs more than six players and less than 39 players, or inputs not valid')
        return
    progress_window = tk.Toplevel(root)
    progress_window.title('Please Wait')
    tk.Label(progress_window, text='In progress, please wait...').pack(padx=20, pady=20)
    progress_window.update()
    top_solutions = []
    team_limit = 5
    num_individuals = 200
    num_generations = 35
    top_n_solutions = generate_lineups
    mutation_rate = 0.05
    lp_solution, captainI = solve_lp()
    population = []
    if lp_solution:
        for _ in range(1500):
            temp = Individual(position=lp_solution, captain=captainI)
            population.append(temp)
        top_solutions.append((temp.position.copy(), temp.fitness, temp.captain))
        print(temp.fitness)
        print(temp.position)
        print(temp.captain)
    while len(population) < num_individuals:
        population.append(Individual())
        population = [Individual() for _ in range(num_individuals)]
    print(len(top_solutions))
    for individual in population:
        if individual.fitness > 0 and is_solution_unique(individual.position, top_solutions):
            if len(top_solutions) < top_n_solutions:
                top_solutions.append((individual.position.copy(), individual.fitness, individual.captain.copy()))
            else:  # inserted
                top_solutions = sorted(top_solutions, key=lambda x: x[1], reverse=True)
                if individual.fitness > top_solutions[(-1)][1]:
                    top_solutions[(-1)] = (individual.position.copy(), individual.fitness, individual.captain.copy())
    for generation in range(num_generations):
        new_population = []
        fitness = np.array([individual.fitness for individual in population])
        for _ in range(num_individuals // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1 = parent1.crossover(parent2)
            child2 = parent2.crossover(parent1)
            child1.mutate()
            child2.mutate()
            new_population.extend([child1, child2])
        if firstflag == 1:
            population = population
            firstflag = 0
            print('first')
        else:  # inserted
            population = new_population
        for individual in population:
            if individual.fitness > 0 and is_solution_unique(individual.position, top_solutions):
                if len(top_solutions) < top_n_solutions:
                    top_solutions.append((individual.position.copy(), individual.fitness, individual.captain.copy()))
                else:  # inserted
                    top_solutions = sorted(top_solutions, key=lambda x: x[1], reverse=True)
                    if individual.fitness > top_solutions[(-1)][1]:
                        top_solutions[(-1)] = (individual.position.copy(), individual.fitness, individual.captain.copy())
    top_solutions = sorted(top_solutions, key=lambda x: x[1], reverse=True)
    for idx, (solution, fitness, captain) in enumerate(top_solutions):
        print(f'Solution {idx + 1}:')
        print(solution)
        print(f'Total FPS: {fitness}\n')
        selected_players = df.iloc[solution]
        selected_players.loc[:, 'Position'] = ['CPTN' if idx == captain else 'FLEX' for idx in selected_players.index]
        total_salary = selected_players['SALARY'].sum()
        captain_salary = df.loc[captain, 'SALARY'] * 0.5
        total_salary += captain_salary - df.loc[captain, 'SALARY']
        print(f'Total Salary: ${total_salary:,.2f}')
        print(f'Total FPS: {fitness}\n')
    progress_window.destroy()

def select_lineup():
    if not lineup_number_entry.get().strip():
        messagebox.showerror('Input Error', 'Please enter a lineup number.')
        return
    try:
        lineup_number = int(lineup_number_entry.get())
        if lineup_number < 1 or lineup_number > len(top_solutions):
            raise ValueError('Lineup number must be between 1 and the number of generated lineups')
        display_table(lineup_number)
    except ValueError:
        messagebox.showerror('Input Error', 'Invalid lineup number')
        return None
    else:  # inserted
        pass

def clear_tree(tree):
    for item in tree.get_children():
        tree.delete(item)

def display_table(lineup_number):
    if 'df' in globals():
        clear_tree(tree)
        for item in tree.get_children():
            tree.delete(item)
        tree['columns'] = list(df.columns) + ['Position']
        tree['show'] = 'headings'
        for col in tree['columns']:
            tree.heading(col, text=col)
            tree.column(col, width=100)
        if lineup_number - 1 < len(top_solutions):
            solution, fitness, captain = top_solutions[lineup_number - 1]
            selected_players = df.iloc[solution].copy()
            selected_players['Position'] = ['CPTN' if idx == captain else 'FLEX' for idx in selected_players.index]
            for index, row in selected_players.iterrows():
                tree.insert('', 'end', values=list(row))
            messagebox.showinfo('Lineup Selected', f'Showing lineup number {lineup_number}')
        else:  # inserted
            messagebox.showerror('Error', f'Lineup number {lineup_number} is out of range.')
    else:  # inserted
        messagebox.showerror('Error', 'No CSV file uploaded')


def download_results():
    if not top_solutions:
        messagebox.showerror('Error', 'No solutions available to download.')
        return

    file_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')],
                                             title='Save Solutions As')
    if not file_path:
        return

    try:
        # Open the file and write to it
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            for idx, (solution, fitness, captain) in enumerate(top_solutions):
                f.write(f'Solution {idx + 1}\n')
                selected_players = df.iloc[solution].copy()

                # Assign positions: Captain (CPTN) and others as FLEX
                selected_players['Position'] = ['CPTN' if idx == captain else 'FLEX' for idx in selected_players.index]

                # Multiply captain's salary by 1.5
                selected_players.loc[captain, 'SALARY'] *= 1.5

                # Reorder to have captain first
                selected_players = pd.concat([selected_players.loc[[captain]], selected_players.drop(captain)])
                total_row = selected_players.select_dtypes(include=[np.number]).sum()
                total_row['Position'] = 'TOTAL'
                total_df = pd.DataFrame(total_row).T

                # Append the total row to the DataFrame
                selected_players = pd.concat([selected_players, total_df], ignore_index=True)
                selected_players.to_csv(f, index=False, encoding='utf-8')
                f.write('\n')

        # Show success message after writing completes
        messagebox.showinfo('Success', f'Top solutions have been saved to {os.path.basename(file_path)}')

    except Exception as e:
        # Catch any exceptions and display error details
        error_details = traceback.format_exc()
        print(error_details)
        messagebox.showerror('Error', f'An error occurred while saving the file: {str(e)}')

    return
root = tk.Tk()
root.title('Lineup Optimizer')
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)
canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
scrollbar_x = ttk.Scrollbar(root, orient='horizontal', command=canvas.xview)
scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
canvas.configure(yscrollcommand=scrollbar.set, xscrollcommand=scrollbar_x.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
second_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=second_frame, anchor='nw')
second_frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
labels = ['Salary (25,000 - 50,000):', 'Daily MatchUp (0 - 500):', 'Consistency (0 - 500):', 'Proj. Ownership (0 - 500):', 'Proj. FPS (0 - 500 or range e.g., \'65 - 75\'):', 'Proj. Ceiling (0 - 500):', 'Proj. Floor (0 - 500):', 'FPS AVG (0 - 500):', 'Leverage (0-500):', 'Safety (0-1000):', 'OPP V POS (0-500):', 'Value (0-500)', 'Floor Value (0-500):', 'Ceiling Value (0-500):', 'Generate Line Ups (1 - 150):']
entries = []
for i, label_text in enumerate(labels):
    label = tk.Label(second_frame, text=label_text)
    label.grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
    entry = tk.Entry(second_frame)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)
salary_entry, matchup_entry, consistency_entry, ownership_entry, fps_entry, ceiling_entry, floor_entry, fps_avg_entry, leverage_entry, safety_entry, opp_v_pos_entry, value_entry, floor_val_entry, ceiling_val_entry, lineups_entry = entries
upload_button = tk.Button(second_frame, text='Upload CSV', command=upload_file)
upload_button.grid(row=len(labels), columnspan=2, pady=10)
optimizer_button = tk.Button(second_frame, text='Start Optimizer', command=start_optimizer)
optimizer_button.grid(row=len(labels) + 1, columnspan=2, pady=10)
lineup_number_label = tk.Label(second_frame, text='Select Lineup Number:')
lineup_number_label.grid(row=len(labels) + 2, column=0, sticky=tk.W, padx=10, pady=5)
lineup_number_entry = tk.Entry(second_frame)
lineup_number_entry.grid(row=len(labels) + 2, column=1, padx=10, pady=5)
select_lineup_button = tk.Button(second_frame, text='Select Lineup', command=select_lineup)
select_lineup_button.grid(row=len(labels) + 3, columnspan=2, pady=10)
download_button = tk.Button(second_frame, text='Download Results', command=download_results)
download_button.grid(row=len(labels) + 4, columnspan=2, pady=10)
tree_frame = tk.Frame(second_frame)
tree_frame.grid(row=len(labels) + 5, columnspan=2, padx=10, pady=10)
tree = ttk.Treeview(tree_frame)
tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar_tree_y = ttk.Scrollbar(tree_frame, orient='vertical', command=tree.yview)
scrollbar_tree_y.pack(side=tk.RIGHT, fill=tk.Y)
scrollbar_tree_x = ttk.Scrollbar(tree_frame, orient='horizontal', command=tree.xview)
scrollbar_tree_x.pack(side=tk.BOTTOM, fill=tk.X)
tree.configure(yscrollcommand=scrollbar_tree_y.set, xscrollcommand=scrollbar_tree_x.set)
root.mainloop()