function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = sum(theta' .* X, 2);
h_m_y = h - y;
h_m_y_sqrt = h_m_y .^ 2;
J = sum(h_m_y_sqrt) / (2 * m);


% =========================================================================

end
