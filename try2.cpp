#include <unordered_map>
#include <tuple>
#include <vector>
#include <functional>
#include <iostream>
#include <cmath>

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
    double beta = 1e-5;  // the time exploration diminshes
    double gamma = 0.9;  // Discount factor
    // Used in computing price range
    double b = 1;     // sensitivity of demand (not set by paper)
    double ksi = 0.1; // possible price range (not set by paper)
    int m = 15;       // discretize the price range to m equally spaced points
    // used in computing demand and reward
    double miu = 0.25; // index of horizontal differentiation
    double a = 2;      // product quality index that capture vertical differntiation
    double c = 1;      // Marginal cost
};

/* Compute hash function */
struct StateActionHash
{
    size_t operator()(const StateAction &sa) const
    {
        // Extract components from the tuple
        double price1 = get<0>(get<0>(sa));
        double price2 = get<1>(get<0>(sa));
        double action = get<1>(sa);

        // Compute individual hash values and combine them using XOR
        size_t hash1 = hash<double>()(price1);
        size_t hash2 = hash<double>()(price2);
        size_t hash3 = hash<double>()(action);

        // Combine the hash values using XOR to ensure a unique hash
        return hash1 ^ (hash2 << 1) ^ (hash3 << 2);
    }
};

struct StateHash {
    size_t operator()(const State &state) const {
        // Extract components from the tuple
        double price1 = get<0>(state);
        double price2 = get<1>(state);

        // Compute individual hash values and combine them using XOR
        size_t hash1 = hash<double>()(price1);
        size_t hash2 = hash<double>()(price2);

        // Combine the hash values using XOR to ensure a unique hash
        return hash1 ^ (hash2 << 1);
    }
};


/* Q table implementation */
// hashed function is used to make the searching faster
unordered_map<StateAction, double, StateActionHash> Q;
double get_q_value(const State &state, double action)
{
    StateAction sa = make_tuple(state, action);
    if (Q.find(sa) == Q.end())
    {
        // change
        Q[sa] = 0.0; // Initialize Q-value to 0 if it doesn't exist
    }
    return Q[sa];
}

void update_q_value(const State &state, double action, double reward, double alpha, double gamma, const State &next_state, vector<double> possible_prices)
{
    double max_future_q = 0.0;
    for (double next_action : possible_prices)
    {
        max_future_q = max(max_future_q, get_q_value(next_state, next_action));
    }
    Q[make_tuple(state, action)] = get_q_value(state, action) + alpha * (reward + gamma * max_future_q - get_q_value(state, action));
}

/* Get price ranges */
vector<double> get_price_range(Game &game)
{
    double p_N = (game.a + game.b) / (2 * game.b - game.miu); // nash equalibrium price
    double p_M = (game.a - game.c) / (2 * game.b);            // monopoly price
    double low_bound = p_N - game.ksi * (p_M - p_N) * 1000;
    double up_bound = p_M + game.ksi * (p_M - p_N) * 1000;
    double step_size = (p_M - p_N) / (game.m - 1);
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
double choose_action(const State &state, double epsilon, vector<double> possible_prices)
{
    if ((double)rand() / RAND_MAX < epsilon)
    {
        // Exploration: choose a random action
        return possible_prices[rand() % possible_prices.size()];
    }
    else
    {
        // Exploitation: choose the best-known action
        double best_action = possible_prices[0];
        double best_q_value = get_q_value(state, best_action);
        for (double action : possible_prices)
        {
            double q_value = get_q_value(state, action);
            if (q_value > best_q_value)
            {
                best_q_value = q_value;
                best_action = action;
            }
        }
        return best_action;
    }
}

/* get reward */
// int agent is an integer (0 or 1) to indicate which player it is
// curr_price is the same as state when k = 1; but this funciton need to be re-implemented when k > 1
float get_reward(State curr_price, int agent, Game &game)
{
    double a_0 = 0; // product 0 is the outside good, so a_0 is an inverse index of aggregate demand
    double denomenator = exp(a_0 / game.miu);
    // get reward for agent 0
    if (agent == 0)
    {
        for (int i = 0; i < 2; i++)
        {
            denomenator += exp((game.a - get<0>(curr_price)) / game.miu);
        }
        double q = exp((game.a - get<0>(curr_price)) / game.miu) / denomenator;
        return (get<0>(curr_price) - game.c) * q;
    }

    // get reward for agent 1
    else
    {
        for (int i = 0; i < 2; i++)
        {
            denomenator += exp((game.a - get<1>(curr_price)) / game.miu);
        }
        double q = exp((game.a - get<1>(curr_price)) / game.miu) / denomenator;
        return (get<1>(curr_price) - game.c) * q;
    }
}

/* Tracking action stability */
struct ConvergenceTracker
{
    unordered_map<State, double, StateHash> current_action;
    unordered_map<State, int, StateHash> stability_counter;
    const int STABILITY_THRESHOLD = 100000;

    // check convergence for the current state
    bool check_and_update_convergence(const State &state, double action)
    {
        // Retrieve or initialize the current action and its stability count
        auto &last_action = current_action[state];
        auto &counter = stability_counter[state];

        // Check if the action has changed
        if (last_action == action)
        {
            counter++;
        }
        else
        {
            last_action = action;
            counter = 1; // Reset the counter
        }

        // Return true if the counter has reached the stability threshold
        return counter >= STABILITY_THRESHOLD;
    }

    bool has_converged()
    {
        // Check if all states have converged (stability counter >= STABILITY_THRESHOLD for all)
        for (const auto &state : stability_counter)
        {
            if (state.second < STABILITY_THRESHOLD)
            {
                return false;
            }
        }
        return true;
    }
};

/* print Q matrix */
void printQMatrix(const unordered_map<StateAction, double, StateActionHash> &Q) {
    int count = 0;
    for (const auto &entry : Q) {
        // Extract the state and action from the entry
        const State &state = get<0>(entry.first);
        double action = get<1>(entry.first);
        double q_value = entry.second;

        // Extract individual prices from the state
        double price1 = get<0>(state);
        double price2 = get<1>(state);

        // Print the state, action, and Q-value
        std::cout << "State (Price1: " << price1 << ", Price2: " << price2 << "), "
                  << "Action: " << action << ", "
                  << "Q-value: " << q_value << std::endl;
                  
        count++;
    }
    cout << "Total num of cells " << count << endl;
}



/* Initialize Q */

/* Simulation */
int main()
{
    // construct a game
    Game game;
    ConvergenceTracker tracker;
    // Initial state with prices for both firms
    State state = make_tuple(1.0, 1.0);
    State curr_price = make_tuple(1.0, 1.0); // curr price has the same implementation as state when k = 1
    // get price range
    vector<double> possible_prices = get_price_range(game);

    long long iterations = 0;
    bool converged = false;

    while (iterations < 10000)
    {
        double epsilon = exp(-game.beta * iterations);
        /***************************** Agent 1's action ************************/
        // Agent 1 chooses an action (price)
        double action1 = choose_action(state, epsilon, possible_prices);

        // Agent 2's current price
        double action2 = get<1>(state);

        // Calculate rewards for both agents
        double reward1 = get_reward(state, 0, game);

        // Next state after taking actions
        State next_state = make_tuple(action1, action2);

        // Update Q-values
        update_q_value(state, action1, reward1, game.alpha, game.gamma, next_state, possible_prices);
        
        // Check convergence for agent 1
        tracker.check_and_update_convergence(state, action1);

        // Move to next state
        state = next_state;

        /******************** Agent 2's action *********************************/
        action1 = get<0>(state);
        action2 = choose_action(state, epsilon, possible_prices);
        double reward2 = get_reward(state, 1, game);
        next_state = make_tuple(action1, action2);
        update_q_value(state, action1, reward1, game.alpha, game.gamma, next_state, possible_prices);
        // Check convergence for agent 2
        tracker.check_and_update_convergence(state, action2);
        if (tracker.has_converged()) {
            cout << "Convergence achieved after " << iterations << " iterations." << endl;
            converged = true;
            break;
        }
        state = next_state;
        iterations++;
    }
    if (!converged){
        cout <<"not converged" << endl;
    }
    printQMatrix(Q);
    return 0;

}
