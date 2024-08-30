Brief Code Explanation:

Only the main.cpp is useful.

try1.cpp and try2.cpp are two unsucessful attempts that should be ignored

Struct game:
- This struct consists of all the initialization of parameters.
- The functions inside the struct are used to compute the price range.
- Implementation adopted from this repository.
- Note: The implementation has not been carefully checked.

Q Matrix:
- The Q matrix is defined as Q[state_index][action].
- state_index is the combination of prices by agent 1 and agent 2, range from 0 to 254 (total m*m states)
- This approach is used to reduce the dimension of the Q matrix and simplify the code.
- For the exact mapping method, please refer to the function state_to_index.

Convergence Tracker:
- This component checks if the Q matrix for agent 1 and agent 2 has converged at the current state and as a whole.

Current Issue:
- The algorithm does not converge at all.
- It seems that the algorithm only converges on state 32. This might be due to an implementation error.

Further Work:
- Plot the price changes of the two agents over time.
