function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;
C1=[0.1,0.1,1,10,0.03,0.3,3,30];
sigma1=[0.1,0.1,1,10,0.03,0.3,3,30];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
error=100000;
for i=1:size(C1,2);
  for j=1:size(sigma1,2);
    model=svmTrain(X, y, C1(1,i), @(x1, x2) gaussianKernel(x1, x2, sigma1(1,j)));
    predict=svmPredict(model,Xval);
    err=mean((predict-=yval).^2.^(1/2));
    if err<error
    error=err;
    C=C1(1,i);
    sigma=sigma1(1,j);
    endif
  end
end

C
sigma



% =========================================================================

end
