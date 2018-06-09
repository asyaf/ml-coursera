function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h = sum(theta' .* X, 2);
h_m_y = h - y;
h_m_y_sqr = h_m_y .^ 2;
J_non_reg = sum(h_m_y_sqr) / (2 * m);
theta_sqr = theta .* theta;
theta_sqr(1) = 0;
J_reg = lambda * sum(theta_sqr) / (2 * m);

J = J_non_reg + J_reg;

% compute for all grad and corret later
theta(1) = 0;
grad_non_reg = X' * h_m_y / m;
grad_reg = lambda * theta / m;

grad = grad_non_reg + grad_reg;
% correct first element
grad(1) = sum(h_m_y) / m;







% =========================================================================

grad = grad(:);

end
