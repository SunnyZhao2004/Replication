#include <unordered_map>
#include <tuple>
#include <vector>
#include <functional>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

// State representation for 2 firms: a pair of prices
using State = tuple<double, double>;

// State-Action pair representation
using StateAction = tuple<State, double>;

// game parameters
struct Game
{
    // used in Q learning implementation
    double alpha = 0.125; // Learning rate
    double beta = 1e-5;   // the time exploration diminshes
    double gamma = 0.9;   // Discount factor
    // Used in computing price range
    double b = 1;     // sensitivity of demand (not set by paper)
    double ksi = 0.1; // possible price range (not set by paper)
    int m = 15;       // discretize the price range to m equally spaced points
    // used in computing demand and reward
    double miu = 0.25; // index of horizontal differentiation
    double a = 2;      // product quality index that capture vertical differntiation
    double c = 1;      // Marginal cost

    // scaler
    int scaler = 1;
};

/* Map state into index */
int state_to_index(double price1, double price2, const vector<double> &possible_prices)
{
    int index1 = -1, index2 = -1;
    for (int i = 0; i < possible_prices.size(); ++i)
    {
        if (abs(price1 - possible_prices[i]) < 1e-5)
            index1 = i;
        if (abs(price2 - possible_prices[i]) < 1e-5)
            index2 = i;
    }
    return index1 * possible_prices.size() + index2;
}

pair<int, int> index_to_state(int index, int m)
{
    int index1 = index / m;
    int index2 = index % m;
    return make_pair(index1, index2);
}

double get_q_value(int state_index, int action_index, const vector<vector<double>> &Q)
{
    return Q[state_index][action_index];
}

void update_q_value(int state_index, int action_index, double reward, double alpha, double gamma, int next_state_index, vector<vector<double>> &Q)
{
    double max_future_q = *max_element(Q[next_state_index].begin(), Q[next_state_index].end());
    Q[state_index][action_index] = Q[state_index][action_index] + alpha * (reward + gamma * max_future_q - Q[state_index][action_index]);
}

/* Get price ranges */
vector<double> get_price_range(Game &game)
{
    double p_N = (game.a + game.b) / (2 * game.b - game.miu); // nash equalibrium price
    double p_M = (game.a - game.c) / (2 * game.b);            // monopoly price
    double low_bound = p_N - game.ksi * (p_M - p_N) * game.scaler;
    double up_bound = p_M + game.ksi * (p_M - p_N) * game.scaler;
    double step_size = (up_bound - low_bound) / (game.m - 1);
    vector<double> possible_prices;
    possible_prices.reserve(game.m); // Reserve space for m elements
    // Fill the array using a single loop
    for (int i = 0; i < game.m; ++i)
    {
        possible_prices.push_back(low_bound + i * step_size);
    }
    return possible_prices;
}

/* Get actions */
double choose_action(int state_index, double epsilon, const vector<vector<double>> &Q, int m)
{
    if ((double)rand() / RAND_MAX < epsilon)
    {
        // Exploration: choose a random action
        return rand() % m;
    }
    else
    {
        // Exploitation: choose the best-known action
        int best_action = 0;
        double best_q_value = Q[state_index][0];
        for (int action = 1; action < m; ++action)
        {
            if (Q[state_index][action] > best_q_value)
            {
                best_q_value = Q[state_index][action];
                best_action = action;
            }
        }
        return best_action;
    }
}

/* get reward */
// int agent is an integer (0 or 1) to indicate which player it is
// curr_price is the same as state when k = 1; but this funciton need to be re-implemented when k > 1
float get_reward(int price_index1, int price_index2, int agent, Game &game, const vector<double> &possible_prices)
{
    double price1 = possible_prices[price_index1];
    double price2 = possible_prices[price_index2];

    double a_0 = 0; // product 0 is the outside good, so a_0 is an inverse index of aggregate demand
    double denomenator = exp(a_0 / game.miu);

    // Calculate reward for agent 0
    if (agent == 0)
    {
        denomenator += exp((game.a - price1) / game.miu) + exp((game.a - price2) / game.miu);
        double q = exp((game.a - price1) / game.miu) / denomenator;
        return (price1 - game.c) * q;
    }
    // Calculate reward for agent 1
    else
    {
        denomenator += exp((game.a - price1) / game.miu) + exp((game.a - price2) / game.miu);
        double q = exp((game.a - price2) / game.miu) / denomenator;
        return (price2 - game.c) * q;
    }
}

struct ConvergenceTracker
{
    vector<int> current_action1;    // Vector to store the current action (price index) for each state for Agent 1
    vector<int> current_action2;    // Vector to store the current action (price index) for each state for Agent 2
    vector<int> stability_counter1; // Vector to store the stability count for each state for Agent 1
    vector<int> stability_counter2; // Vector to store the stability count for each state for Agent 2
    const int STABILITY_THRESHOLD = 1000;

    ConvergenceTracker() = default;
    ConvergenceTracker(int num_states)
    {
        current_action1.resize(num_states, -1);   // Initialize with a default invalid action index (-1)
        current_action2.resize(num_states, -1);   // Initialize with a default invalid action index (-1)
        stability_counter1.resize(num_states, 0); // Initialize stability counters to 0 for Agent 1
        stability_counter2.resize(num_states, 0); // Initialize stability counters to 0 for Agent 2
    }

    // Check convergence for the current state for Agent 1
    bool check_and_update_convergence1(int state_index, int action_index)
    {
        // Retrieve the current action and its stability count for the given state for Agent 1
        int &last_action = current_action1[state_index];
        int &counter = stability_counter1[state_index];

        // Check if the action has changed
        if (last_action == action_index)
        {
            counter++;
        }
        else
        {
            last_action = action_index;
            counter = 1; // Reset the counter
        }

        // Return true if the counter has reached the stability threshold
        return counter >= STABILITY_THRESHOLD;
    }

    // Check convergence for the current state for Agent 2
    bool check_and_update_convergence2(int state_index, int action_index)
    {

        // Retrieve the current action and its stability count for the given state for Agent 2
        int &last_action = current_action2[state_index];
        int &counter = stability_counter2[state_index];

        // Check if the action has changed
        if (last_action == action_index)
        {
            counter++;
        }
        else
        {
            last_action = action_index;
            counter = 1; // Reset the counter
        }

        // Return true if the counter has reached the stability threshold
        return counter >= STABILITY_THRESHOLD;
    }

    // Check if Agent 1 has converged
    bool has_converged1()
    {
        for (int counter : stability_counter1)
        {
            if (counter < STABILITY_THRESHOLD)
            {
                return false;
            }
        }
        return true;
    }

    // Check if Agent 2 has converged
    bool has_converged2()
    {
        for (int counter : stability_counter2)
        {
            if (counter < STABILITY_THRESHOLD)
            {
                return false;
            }
        }
        return true;
    }

    // Check if both agents have converged
    bool has_converged()
    {
        return has_converged1() && has_converged2();
    }
};

/* Initialize Q */
vector<vector<double>> initialize_q_matrix(int num_states, int m)
{
    return vector<vector<double>>(num_states, vector<double>(m, 0.0));
}

/* Simulation */
void simulate(int state_index){
    // Construct a game
    Game game;
    // Get price range
    vector<double> possible_prices = get_price_range(game);
    int num_states = game.m * game.m;
    ConvergenceTracker tracker(num_states);

    // Initialize Q matrix
    vector<vector<double>> Q1 = initialize_q_matrix(num_states, game.m);
    vector<vector<double>> Q2 = initialize_q_matrix(num_states, game.m);
    long long iterations = 0;
    bool converged = false;
    while (iterations < 10000000)
    {
        double epsilon = exp(-game.beta * iterations);

        /***************************** Agent 1's action ************************/
        // Agent 1 chooses an action (price index)
        int action_index1 = choose_action(state_index, epsilon, Q1, game.m);

        // Agent 2's current price index
        auto [state_price1_index, state_price2_index] = index_to_state(state_index, game.m);
        int action_index2 = state_price2_index;

        // Calculate reward for Agent 1
        double reward1 = get_reward(state_price1_index, state_price2_index, 0, game, possible_prices);
        // Determine the next state index after Agent 1's action
        int next_state_index = state_to_index(possible_prices[action_index1], possible_prices[action_index2], possible_prices);

        // Update Q-values for Agent 1
        update_q_value(state_index, action_index1, reward1, game.alpha, game.gamma, next_state_index, Q1);

        // Check convergence for Agent 1
        if (tracker.check_and_update_convergence1(state_index, action_index1))
        {
            // cout << "Agent 1 has converged for state " << state_index << endl;
        }

        // Update the state index to the new state after Agent 1's action
        state_index = next_state_index;

        /***************************** Agent 2's action ************************/
        action_index2 = choose_action(state_index, epsilon, Q2, game.m);

        // Calculate reward for Agent 2
        double reward2 = get_reward(action_index1, action_index2, 1, game, possible_prices);

        // Determine the next state index after Agent 2's action
        next_state_index = state_to_index(possible_prices[action_index1], possible_prices[action_index2], possible_prices);

        // Update Q-values for Agent 2
        update_q_value(state_index, action_index2, reward2, game.alpha, game.gamma, next_state_index, Q2);

        // After updating Q2 for Agent 2
        if (tracker.check_and_update_convergence2(state_index, action_index2))
        {
            // cout << "Agent 2 has converged for state " << state_index << endl;
        }
        if (tracker.has_converged())
        {
            cout << "Convergence achieved after " << iterations << " iterations." << endl;
            converged = true;
            break;
        }

        // Update the state index to the new state after Agent 2's action
        state_index = next_state_index;

        iterations++;
    }

    if (!converged)
    {
        cout << "Not converged after " << iterations << " iterations." << endl;
    }

    // Optionally, print the Q matrix for debugging purposes
    for (int i = 0; i < num_states; ++i)
    {
        auto [state_price1_index, state_price2_index] = index_to_state(i, game.m);
        cout << "State (Price1 Index: " << state_price1_index << ", Price2 Index: " << state_price2_index << ") -> Q-values: ";
        for (int j = 0; j < game.m; ++j)
        {
            cout << Q1[i][j] << " ";
        }
        cout << endl;
    }
    
}

/* Simulation */
int main()
{
    simulate(100);
    return 0;
}