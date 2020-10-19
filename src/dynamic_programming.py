import numpy as np
from maze import build_maze
from toolbox import random_policy


def get_policy_from_q(q) :
    m,n=q.shape
    policy=[]
    for i in range(m): 
        policy.append(np.argmax(q[i,:]))
    return policy
   
def get_policy_from_v(mdp, v) :
    # Outputs a policy given the state values
    policy = np.zeros(mdp.nb_states)  # initial state values are set to 0
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        v_temp = []
        for u in mdp.action_space.actions:
            if x not in mdp.terminal_states:
                # Process sum of the values of the neighbouring states
                summ = 0
                for y in range(mdp.nb_states):
                    summ = summ + mdp.P[x, u, y] * v[y]
                v_temp.append(mdp.r[x, u] + mdp.gamma * summ)
            else:  # if the state is final, then we only take the reward into account
                v_temp.append(mdp.r[x, u])
        policy[x] = np.argmax(v_temp)
    return policy


def improve_policy_from_v(mdp, v, policy):
    # Improves a policy given the state values
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        v_temp = np.zeros(mdp.action_space.size)
        for u in mdp.action_space.actions:
            if x not in mdp.terminal_states:
                # Process sum of the values of the neighbouring states
                summ = 0
                for y in range(mdp.nb_states):
                    summ = summ + mdp.P[x, u, y] * v[y]
                v_temp[u] = mdp.r[x, u] + mdp.gamma * summ
            else:  # if the state is final, then we only take the reward into account
                v_temp[u] = mdp.r[x, u]

        for u in mdp.action_space.actions:
            if v_temp[u] > v_temp[policy[x]]:
                policy[x] = u
    return policy


def evaluate_one_step_v(mdp, v, policy):
    # Outputs the state value function after one step of policy evaluation
    # Corresponds to one application of the Bellman Operator
    v_new = np.zeros(mdp.nb_states)  # initial state values are set to 0
    for x in range(mdp.nb_states):  # for each state x
        # Compute the value of the state x for each action u of the MDP action space
        v_temp = []
        if x not in mdp.terminal_states:
            # Process sum of the values of the neighbouring states
            summ = 0
            for y in range(mdp.nb_states):
                summ = summ + mdp.P[x, policy[x], y] * v[y]
            v_temp.append(mdp.r[x, policy[x]] + mdp.gamma * summ)
        else:  # if the state is final, then we only take the reward into account
            v_temp.append(mdp.r[x, policy[x]])

        # Select the highest state value among those computed
        v_new[x] = np.max(v_temp)
    return v_new


def evaluate_v(mdp, policy):
    # Outputs the state value function of a policy
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    stop = False
    while not stop:
        vold = v.copy()
        v = evaluate_one_step_v(mdp, vold, policy)

        # Test if convergence has been reached
        if (np.linalg.norm(v - vold)) < 0.01:
            stop = True
    return v

'''
def evaluate_one_step_q(mdp, q, policy) :
    # Outputs the state value function after one step of policy evaluation
    # TODO : fill this


def evaluate_q(mdp, policy) :
    # Outputs the state value function of a policy
    # TODO : fill this'''
  
# ------------------------- Value Iteration with the V function ----------------------------#
# Given a MDP, this algorithm computes the optimal state value function V
# It then derives the optimal policy based on this function
# This function is given


def value_iteration_v(mdp, render=True):
    # Value Iteration using the state value v
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    stop = False

    if render:
        mdp.new_render()

    while not stop:
        v_old = v.copy()
        if render:
            mdp.render(v)

        for x in range(mdp.nb_states):  # for each state x
            # Compute the value of the state x for each action u of the MDP action space
            v_temp = []
            for u in mdp.action_space.actions:
                if x not in mdp.terminal_states:
                    # Process sum of the values of the neighbouring states
                    summ = 0
                    for y in range(mdp.nb_states):
                        summ = summ + mdp.P[x, u, y] * v_old[y]
                    v_temp.append(mdp.r[x, u] + mdp.gamma * summ)
                else:  # if the state is final, then we only take the reward into account
                    v_temp.append(mdp.r[x, u])

                    # Select the highest state value among those computed
            v[x] = np.max(v_temp)

        # Test if convergence has been reached
        if (np.linalg.norm(v - v_old)) < 0.01:
            stop = True

    policy = get_policy_from_v(mdp, v)
    if render:
        mdp.render(v, policy)

    return v

# ------------------------- Value Iteration with the Q function ----------------------------#
# Given a MDP, this algorithm computes the optimal action value function Q
# It then derives the optimal policy based on this function


def value_iteration_q(mdp, render=True):
    q = np.zeros((mdp.nb_states, mdp.action_space.size))  # initial action values are set to 0
    stop = False

    if render:
        mdp.new_render()

    while not stop:
        qold = q.copy()

        if render:
            mdp.render(q)

        for x in range(mdp.nb_states):
            for u in mdp.action_space.actions:
                if x in mdp.terminal_states:
                    q[x, u] = r[x,u]
                else:
                    summ = 0
                    q_temp = 0
                    for y in range(mdp.nb_states): 
                        q_temp=np.max(q[y,:])   
                        summ=summ+mdp.P(x,u,y)*q_temp  
            q[x, u] = r[x,u]+mdp.gamma*summ
        if (np.linalg.norm(q - qold)) <= 0.01:
            stop = True

    if render:
        mdp.render(q)
    return q


# ------------------------- Policy Iteration with the Q function ----------------------------#
# Given a MDP, this algorithm simultaneously computes the optimal action value function Q and the optimal policy

def policy_iteration_q(mdp, render=True):  # policy iteration over the q function
    q = np.zeros((mdp.nb_states, mdp.action_space.size))  # initial action values are set to 0
    policy = random_policy(mdp)

    stop = False

    if render:
        mdp.new_render()

    while not stop:
        qold = q.copy()

        if render:
            mdp.render(q)

        # Step 1 : Policy evaluation
        # TODO : fill this

        # Step 2 : Policy improvement
        # TODO : fill this

        # Check convergence
        if (np.linalg.norm(q - qold)) <= 0.01:
            stop = True

    if render:
        mdp.render(q, get_policy_from_q(q))
    return q


# ------------------------- Policy Iteration with the V function ----------------------------#
# Given a MDP, this algorithm simultaneously computes the optimal state value function V and the optimal policy

def policy_iteration_v(mdp, render=True):
    # policy iteration over the v function
    v = np.zeros(mdp.nb_states)  # initial state values are set to 0
    policy = random_policy(mdp)

    stop = False

    if render:
        mdp.new_render()

    while not stop:
        vold = v.copy()
        # Step 1 : Policy Evaluation
        # TODO : fill this

        if render:
            mdp.render(v)
            mdp.plotter.render_pi(policy)

        # Step 2 : Policy Improvement
        # TODO : fill this

        # Check convergence
        if (np.linalg.norm(v - vold)) < 0.01:
            stop = True

    if render:
        mdp.render(v)
        mdp.plotter.render_pi(policy)
    return v


def run_dyna_prog():
    # walls = [14, 15, 16, 31, 45, 46, 47]
    # height = 6
    # width = 9
    walls = [5, 6, 13]
    height = 4
    width = 5

    m = build_maze(width, height, walls)  # maze-like MDP definition
    input("press enter")


    print("value iteration V")
    value_iteration_v(m, render=True)
    print("value iteration Q")
    value_iteration_q(m, render=True)
    #print("policy iteration Q")
    #policy_iteration_q(m, render=True)
    #print("policy iteration V")
    #policy_iteration_v(m, render=True)
    
    

if __name__ == '__main__':
    run_dyna_prog()
