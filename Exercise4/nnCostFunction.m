function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 


% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Forward propagation
% a1 = X
X = [ones(m,1) X]; % adding bias unit 

z2 = Theta1 * X';  % (25*401)*(401*5000)
a2 = sigmoid(z2); % (25*5000)

a2 = [ones(m,1) a2']; % adding bias unit 
z3 = Theta2 * a2';

hypo_theta = sigmoid(z3); % hypo_theta = a3

% Recode labels as vectors containing only values 0 or 1
new_y = zeros(num_labels, m); % 10*5000
for i = 1:m,                 % loop through every training example (i corresponding to index col)
    new_y(y(i), i) = 1;      % mark the corresponding y of the training example in the new y as 1
end

J = (1/m) * sum( sum ( ( -new_y .* log(hypo_theta) ) - ( (1 - new_y) .* log(1 - hypo_theta ) ) ) );

% Regularization
% Removing the bias term
t1 = Theta1(:,2:size(Theta1,2));
t2 = Theta2(:,2:size(Theta2,2));

Reg = lambda  * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 ))) / (2*m);

% Regularized cost function
J = J + Reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

% Back propagation
for t=1:m

    % Step 1 - a1(t) = X(t)
	a1 = X(t,:); % X already have a bias (1*401)
    a1 = a1'; % (401*1)

    % Step 2
	z2 = Theta1 * a1; % (25*401)*(401*1)
	a2 = sigmoid(z2); % (25*1)
    
    a2 = [1 ; a2]; % adding a bias (26*1)
	z3 = Theta2 * a2; % (10*26)*(26*1)
	a3 = sigmoid(z3); % final activation layer a3 == h(theta) (10*1)
    
    % Step 3 - 𝛿
	delta_3 = a3 - new_y(:,t); % (10*1)
    z2=[1; z2]; % bias (26*1)

    % Step 4
    delta_2 = (Theta2' * delta_3) .* sigmoidGradient(z2); % ((26*10)*(10*1))=(26*1)
	delta_2 = delta_2(2:end); % skipping sigma2(0) (25*1)

    %Step 5.1
	Theta2_grad = Theta2_grad + delta_3 * a2'; % (10*1)*(1*26)
	Theta1_grad = Theta1_grad + delta_2 * a1'; % (25*1)*(1*401)
    
end;

% Step 5.2
Theta2_grad = (1/m) * Theta2_grad; % (10*26) - used to set j = 0
Theta1_grad = (1/m) * Theta1_grad; % (25*401) - used to set j = 0

% Regularization
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1

% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end