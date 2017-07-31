function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n=size(theta,1);
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
temp=theta;
for T_iter=1:n;
  sum=0;
  for i=1:m,
    hypo=0;
    for j=1:n,
      hypo=hypo+(theta(j,1)*X(i,j));
    end
    sum=sum+((hypo-y(i,1))*X(i,T_iter));
  end
  temp(T_iter,1)=theta(T_iter,1)-(alpha*(sum)/m);
end

theta=temp;



    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
