function [loss] = loss_function(portfolio_returns, VaR, alpha)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    T = size(portfolio_returns, 1);
    h = std(portfolio_returns) * T^(-1.5);
    sum_loss = 0;
    for i = 1:T
        loss = normcdf( (portfolio_returns(i) - VaR)/h );
        sum_loss = sum_loss + loss;
    end
    average_loss = sum_loss / T;
    loss = (average_loss - alpha).^2;
end