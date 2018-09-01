function y = logisticSigmoid(x)
% simpleLogisticSigmoid Logistic sigmoid activation function
% 
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector where the logistic sigmoid was applied element by
% element.
%
     %y=2/(1+exp(-2*x)) -1;
    y = 1./(1 + exp(-x));
   %y=tanh(x);
end