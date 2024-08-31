#include <unordered_map>
#include <tuple>
#include <vector>
#include <functional>
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace std;

// game parameters
struct Game {

    // used in Q learning implementation
    double alpha = 0.125; // Learning rate
    double beta = 1e-5;   // the time exploration diminshes
    double gamma = 0.9;   // Discount factor
    // Used in computing price range
    double b = 1;     // sensitivity of demand (not set by paper)
    double ksi = 0.1; // possible price range (not set by paper)
    int m = 15;       // discretize the price range to m equally spaced points
    // used in computing demand and reward
    double mu = 0.25; // index of horizontal differentiation
    double a = 2;      // product quality index that capture vertical differntiation
    double c = 1;      // Marginal cost

    // Demand function for one product, considering the price of both products
    float get_demand(double price1, double price2, int product_index) const {
        double a_0 = 0;  // Product 0 is the outside good, so a_0 is an inverse index of aggregate demand
        double denominator = exp(a_0 / mu);
        denominator += exp((a - price1) / mu) + exp((a - price2) / mu);

        // Compute the demand for the specified product
        if (product_index == 0) {
            return exp((a - price1) / mu) / denominator;
        } else {
            return exp((a - price2) / mu) / denominator;
        }
    }

    // First-order condition for Nash equilibrium
    double foc(double price1, double price2) const {
        float demand = get_demand(price1, price2, 0);
        double zero = 1.0 - (price1 - c) * (1.0 - demand) / mu;
        return zero;
    }

    // First-order condition for monopoly
    double foc_monopoly(double price1, double price2) const {
        double demand0 = get_demand(price1, price2, 0);
        double demand1 = get_demand(price1, price2, 1);
        double zero = 1.0 - (price1 - c) * (1.0 - demand0) / mu + (price2 - c) * demand1 / mu;
        return zero;
    }

    // Function to solve for the price using Newton's method
    double solve_foc(const function<double(double)>& foc_func, double p0) const {
        const double tolerance = 1e-6;
        const int max_iter = 100;
        double p = p0;

        for (int i = 0; i < max_iter; ++i) {
            double F = foc_func(p);
            double dF_dp = (foc_func(p + 1e-6) - F) / 1e-6;  // Numerical derivative

            if (fabs(dF_dp) < tolerance) {
                cerr << "Derivative too small, stopping iteration." << endl;
                break;
            }

            double delta_p = -F / dF_dp;
            p += delta_p;

            if (fabs(delta_p) < tolerance) {
                break;
            }
        }

        return p;
    }

    // Compute the price range, including competitive (Nash equilibrium) and monopoly prices
    vector<double> get_price_range() const {
        double p0 = 3.0 * c;  // Initial guess for price

        // Solve for Nash equilibrium (competitive) price
        double p_N = solve_foc([this](double p) { return foc(p, p); }, p0);

        // Solve for monopoly price
        double p_M = solve_foc([this](double p) { return foc_monopoly(p, p); }, p0);

        // Calculate the price range
        double low_bound = p_N - ksi * (p_M - p_N);
        double up_bound = p_M + ksi * (p_M - p_N) ;
        double step_size = (up_bound - low_bound) / (m - 1);

        vector<double> possible_prices;
        possible_prices.reserve(m); // Reserve space for m elements

        // Fill the array using a single loop
        for (int i = 0; i < m; ++i) {
            possible_prices.push_back(low_bound + i * step_size);
        }

        return possible_prices;
    }
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

void update_q_value(int state_index, int action_index, double reward, double alpha, double gamma, int next_state_index, vector<vector<double>> &Q)
{
    double max_future_q = *max_element(Q[next_state_index].begin(), Q[next_state_index].end());
    Q[state_index][action_index] = (1 - alpha) * Q[state_index][action_index] + alpha * (reward + gamma * max_future_q);
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

/* get demand */
// int agent is an integer (0 or 1) to indicate which player it is
float get_demand(const Game &game, double price1, double price2, int agent)
{
    double a_0 = 0; // product 0 is the outside good, so a_0 is an inverse index of aggregate demand
    double denomenator = exp(a_0 / game.mu) + exp((game.a - price1) / game.mu) + exp((game.a - price2) / game.mu);
    double q;
    if (agent == 0)
    {
        q = exp((game.a - price1) / game.mu) / denomenator;
    }
    else
    {
        q = exp((game.a - price2) / game.mu) / denomenator;
    }
    return q;
}

/* get reward */
// int agent is an integer (0 or 1) to indicate which player it is
// curr_price is the same as state when k = 1; but this funciton need to be re-implemented when k > 1
float get_reward(int price_index1, int price_index2, int agent, const Game &game, const vector<double> &possible_prices)
{
    double price1 = possible_prices[price_index1];
    double price2 = possible_prices[price_index2];

    double a_0 = 0; // product 0 is the outside good, so a_0 is an inverse index of aggregate demand
    double denomenator = exp(a_0 / game.mu);

    // Calculate reward for agent 0
    if (agent == 0)
    {
        float q = get_demand(game, price1, price2, 0);
        return (price1 - game.c) * q;
    }
    // Calculate reward for agent 1
    else
    {
        float q = get_demand(game, price1, price2, 1);
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
void simulate(int state_index)
{
    // Construct a game
    Game game;
    // Get price range
    vector<double> possible_prices = game.get_price_range();
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
            cout << "Agent 1 has converged for state " << state_index << endl;
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
            cout << "Agent 2 has converged for state " << state_index << endl;
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

    // print the Q matrix for debugging purposes
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
    // choose a random index to start
    simulate(4);
    return 0;
}