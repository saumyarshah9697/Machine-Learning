function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

n=(size(X)(2));
%t=length(theta);
for i=1:m,
  hypo=0;
  for j=1:n,
    hypo=hypo+theta(j,1)*X(i,j);
  end
  J=J+((hypo-y(i,1))^2);
end
J=J/(2*m);



% =========================================================================

end
