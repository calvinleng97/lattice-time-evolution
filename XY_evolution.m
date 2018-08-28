%% Introduction
% Authors: Calvin Leng, Han Li, Juan Medina
%
% Description: Runs an animation of the time evolution of a K-particle
% system on an L-sized one dimensional lattice according to the XY-model.
% 
% Time Complexity (Pre-animation computations): O((L choose K)^3)
%     - dominated by matrix exponential computation
% Time Complexity (Per timestep during animation): O(K * (L choose K))



%% Global Macros and Debugging Options
DT = 0.1;
TRACK_RUNTIME = true;

%% User Inputs
L = input('How many lattice points? ');
K = input('How many particles? ');
t_max = input('How many steps would you like to run the animation? ');

start_positions = zeros(1, K);

for i = 1:K
    start_positions(1, i) = input(['Position of ' num2str(i) ...
    'th particle? (must be between 1 and ' num2str(L) ' inclusive): ']);
end

start_positions = sort(start_positions);

%% Creates Ordering F: int -> tuples and F^-1: tuples -> int
% Creates the ordering because there's no natural ordering of 
% k-tuples. This will create a map like so: (example of 5 lattice points
% with 3 particles):
% F: int -> int matrix size 1 x K
% 1 -> [1 2 3];   2 -> [1 2 4];    3 -> [1 2 5];    4 -> [1 3 4];
% 5 -> [1 3 5];   6 -> [1 4 5];    7 -> [2 3 4];    8 -> [2 3 5];
% 9 -> [2 4 5];   10 -> [3 4 5];
%
% F^-1: string -> int
% '[1 2 3]' -> 1;   '[1 2 4]' -> 2;   '[1 2 5]' -> 3;    '[1 3 4]' -> 4;
% '[1 3 5]' -> 5;   '[1 4 5]' -> 6;   '[2 3 4]' -> 7;    '[2 3 5]' -> 8;
% '[2 4 5]' -> 9;   '[3 4 5]' -> 10;
% 
% Time Complexity: O(k * (l choose k))

if TRACK_RUNTIME
    'Constructing ordering map...'
    init_t = cputime;
    init_order_t = cputime;
end

% Precomputed dimension of H^(k) 
lCk = nchoosek(L, K);

% Order mapping function F: int -> int array implemented as a MATLAB list
order_mapping = cell(1, lCk); 

% First image of F, which would be [1 2 3 ... k], implemented as int array
current_order = 1:K;

% Creates inverse mapping F^-1. Uses hash tables with hashing function that
% takes int array, converts it to a string literally, then uses it as a key
inverse_mapping = containers.Map('KeyType', 'char', 'ValueType', 'double');

for i = 1:lCk
    % Inserts an explicit bijection pair: i <-> current_order
    order_mapping{i} = current_order;
    inverse_mapping(mat2str(current_order)) = i;
    
    % Move current_order to the next order.
    % ex (L = 5; K = 3)
    %    [1 2 3] -> [1 2 4]
    %    [1 2 5] -> [1 3 4]
    %    [2 4 5] -> [3 4 5]
    UPDATING_ORDER = true;
    c = K;
    
    while UPDATING_ORDER && c > 0
        if c == K
            if current_order(1, K) + 1 <= L
                current_order(1, K) = current_order(1, K) + 1;
                UPDATING_ORDER = false;
            else
                c = c - 1;
            end
        else
            if current_order(1, c) + 1 < current_order(1, c + 1)
                current_order(1, c) = current_order(1, c) + 1;
                for j = c+1:K
                    current_order(1, j) = current_order(1, j - 1) + 1;
                end
                UPDATING_ORDER = false;
            else
                c = c - 1;
            end
        end
    end
end

if TRACK_RUNTIME
    end_order_t = cputime;
    'Ordering map constructed.'
end

%% Fills the H_k matrix 
% If we define the distance between two tuples as
% the sum of the differences of each slot of the tuples, then
% the matrix is only nonzero when this distance = 1. This loops through
% the rows with the iterating variable `i`. Then, it maps `i` to its
% associated tuple F(i). Then, it goes through the tuple `F(i)` and
% increases or decreases each slot by 1, which we call F(i)'. Then, we find
% its inverse mapping F^-1(F(i)') which we call `j`. Then, we will place a
% 0.5 at the row `i`, column `j`. There is a proof for why this
% is the valid H^(k).
%
% Time Complexity: O(k * (l choose k)))
if TRACK_RUNTIME
    'Constructing H^(k) matrix...'
    init_hk_t = cputime;
end

row_vector = ones(2 * K * lCk, 1);
col_vector = ones(2 * K * lCk, 1);
values_vector = 0.5 * ones(2 * K * lCk, 1);
c = 1;

for i = 1:lCk
    tuple = order_mapping{i};
    for j = 1:K
        pos = tuple(1, j);
        if pos - 1 > 0 && (j == 1 || pos - 1 > tuple(1, j - 1))
            new_tuple = tuple;
            new_tuple(1, j) = new_tuple(1, j) - 1;
            row_vector(c, 1) = i;
            col_vector(c, 1) = inverse_mapping(mat2str(new_tuple));
            c = c + 1;
        end
        if pos + 1 <= L && (j == K || pos + 1 < tuple(1, j + 1))
            new_tuple = tuple;
            new_tuple(1, j) = new_tuple(1, j) + 1;
            row_vector(c, 1) = i;
            col_vector(c, 1) = inverse_mapping(mat2str(new_tuple));
            c = c + 1;
        end
    end
end

% Direct construction of sparse matrix with 3 column vectors.
H_k = sparse(row_vector, col_vector, values_vector);
H_k(1, 1) = 0;

if TRACK_RUNTIME
    end_hk_t = cputime;
    'H^(k) matrix constructed.'
end

%% Compute Phi_0
% Finds the corresponding index for the tuple of starting positions and
% sets that row of Phi_0 to be 1.
%
% Time Complexity: Effectively O(1) as `zeros` is very fast.
Phi_0 = zeros(lCk, 1);
Phi_0(inverse_mapping(mat2str(start_positions)), 1) = 1;

%% Compute exp(-i * DT * H)
% The most time consuming part of the program.
%
% Time Complexity: O((l choose k)^3)
if TRACK_RUNTIME
    'Computing matrix exponential...'
    init_exp_t = cputime;
end

D_EXP = sparse(expm(-1i * DT * H_k));

if TRACK_RUNTIME
    end_exp_t = cputime;
    'Matrix exponential constructed.'
end

%% Debugging results
if TRACK_RUNTIME
    total_t = cputime - init_t;
    order_t = end_order_t - init_order_t;
    hk_t = end_hk_t - init_hk_t;
    exp_t = end_exp_t - init_exp_t;
    ['Runtime results (in seconds): ' char(10) ...
     'Ordering Construction: ' num2str(order_t) char(10) ...
     'H^(k) Construction: ' num2str(hk_t) char(10) ...
     'Matrix Exponential Computation: ' num2str(exp_t) char(10) ...
     'Total Runtime: ' num2str(total_t)]
end

%% Start animation
% Calculates the probability by going through each tuple, finding the
% corresponding entry in Phi, squares it, and adds that to the probability
% of a particle being at that lattice point.
%
% Time Complexity: O(k * (l choose k))

t = 0;
Phi = Phi_0;
Prob = zeros(1, L);

for i = 1:t_max
    % Probability computation
    Prob = zeros(1, L);
    for j = 1:lCk
        tuple = order_mapping{j};
        p = abs(Phi(j, 1))^2;
    
        for m = 1:K
            Prob(1, tuple(1, m)) = Prob(1, tuple(1, m)) + p;
        end
    end
    
    pause(0.01);
    bar(Prob);
    ylim([0 1]);
    t = t + DT;
    Phi = D_EXP * Phi;
end

