#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm> // Include this for max_element

using namespace std;

struct Game
{
    double alpha = 0.15; // Learning rate
    double beta = 4e-6;  // the time exploration diminshes
    double gamma = 0.9;  // Discount factor
    double b = 1;
    double ksi = 0.1;
    int m = 15;
    double miu = 0.25;
    double a = 2;
    double c = 1;
};

vector<double> get_price_range(Game &game)
{
    double p_N = (game.a + game.b) / (2 * game.b - game.miu) * 100;
    double p_M = (game.a - game.c) / (2 * game.b) * 100;
    double low_bound = p_N - game.ksi * (p_M - p_N);
    double up_bound = p_M + game.ksi * (p_M - p_N);
    double step_size = (up_bound - low_bound) / (game.m - 1);
    vector<double> possible_prices(game.m);
    for (int i = 0; i < game.m; ++i)
    {
        possible_prices[i] = low_bound + i * step_size;
    }
    return possible_prices;
}

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

void update_q_value(int state_index, int action_index, double reward, double alpha, double gamma, int next_state_index, vector<vector<double>> &Q)
{
    double max_future_q = *max_element(Q[next_state_index].begin(), Q[next_state_index].end());
    Q[state_index][action_index] = Q[state_index][action_index] + alpha * (reward + gamma * max_future_q - Q[state_index][action_index]);
}

int main()
{
    Game game;
    vector<double> possible_prices = get_price_range(game);
    int num_states = game.m * game.m;

    // Initialize Q matrix
    vector<vector<double>> Q(num_states, vector<double>(game.m, 0.0));

    // Example usage
    double price1 = possible_prices[0];
    double price2 = possible_prices[1];
    int state_index = state_to_index(price1, price2, possible_prices);

    double epsilon = 0.1;
    int action_index = choose_action(state_index, epsilon, Q, game.m);

    // Assume some reward and next state for the example
    double reward = 10.0;
    int next_state_index = state_index; // Simplified for this example
    update_q_value(state_index, action_index, reward, game.alpha, game.gamma, next_state_index, Q);

    // Printing Q matrix for debugging
    int count = 0;
    for (int i = 0; i < num_states; ++i)
    {
        pair<int, int> state_indices = index_to_state(i, game.m);
        int state_price1_index = state_indices.first;
        int state_price2_index = state_indices.second;

        cout << "State (Price1 Index: " << state_price1_index << ", Price2 Index: " << state_price2_index << ") -> Q-values: ";
        for (int j = 0; j < game.m; ++j)
        {
            cout << Q[i][j] << " ";
            count++;
        }
        cout << endl;
    }
    cout << "total num" << count << endl;
    cout << "Possible Prices:" << endl;
    for (double price : possible_prices) {
        cout << price << endl;
    }
    return 0;
}
