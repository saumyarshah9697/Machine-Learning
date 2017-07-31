function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n=length(theta);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sum1=0;
for i=1:m;
  h=sigmoid(X(i,:)*theta);
  sum1=sum1+((y(i)*log(h))+((1-y(i))*log(1-h)));
end
sum2=0;
for i=2:n;
  sum2=sum2+(theta(i)^2);
end
J=(-1*(sum1)/m)+(lambda*(sum2)/(2*m));


term=0;
for i=1:m;
   h=sigmoid(X(i,:)*theta);
  term=term+((h-y(i))*X(i,1));
end
grad(1)=(term)/m;

for j=2:n;
term=0;
  for i=1:m;
   h=sigmoid(X(i,:)*theta);
   term=term+((h-y(i))*X(i,j));
  end
grad(j)=(term/m)+(lambda*theta(j)/m);
end





% =============================================================

end
