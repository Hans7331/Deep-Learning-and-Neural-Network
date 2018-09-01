function y = dLogisticSigmoid(x)
% dLogisticSigmoid Derivative of the logistic sigmoid.
% 
% INPUT:
% x     : Input vector.
%
% OUTPUT:
% y     : Output vector where the derivative of the logistic sigmoid was
% applied element by element.
%

     
    %y = logisticSigmoid(atanh(x)).*(1 - logisticSigmoid(atanh(x)));
    y= 1- (tanh(x)).^2;
end