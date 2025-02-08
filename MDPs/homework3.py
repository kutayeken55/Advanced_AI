import mdp
import mdp_utils
import numpy as np



#Create homework grid world 
grid_env = mdp.gen_simple_world()

#visualize rewards
print("rewards")
mdp_utils.print_array_as_grid(grid_env.rewards, grid_env)

print("visualize a random policy")
random_pi = mdp_utils.get_random_policy(grid_env)
mdp_utils.visualize_policy(random_pi, grid_env)


##TODO: Implement Value iteration then uncomment lines below 
print("--- value iteration ---")
V_vi = mdp_utils.value_iteration(grid_env)
print("Values from Value Iteration")
mdp_utils.print_array_as_grid(V_vi, grid_env)
# ##TODO: implement policy extraction from optimal values
opt_pi = mdp_utils.extract_optimal_policy(V_vi, grid_env)
print("Optimal Policy")
mdp_utils.visualize_policy(opt_pi, grid_env)




# #TODO: Implement Policy Iteration then uncomment the lines below
print("--- policy evaluation ---")
policy_pi, V_pi = mdp_utils.policy_iteration(grid_env)
print("Optimal Policy")
mdp_utils.visualize_policy(policy_pi, grid_env)
print("Values from Policy Iteration")
mdp_utils.print_array_as_grid(V_pi, grid_env)

