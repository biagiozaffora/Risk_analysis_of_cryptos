%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Analysis of risks measures applied to cryptocurrencies         %
%                                                                %
% B. Zaffora           %
%                                                                %
% Financial Econometric : Pr. O. Scaillet                        %
%                                                                %
% Academic year 2019/2020, Fall semester                         %
%                                                                %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% We are interested in the major cryptocurrencies in terms of
% market capitalization, with at least 4 years of history.
% The selected currencies are :
%   BTC = Bitcoin
%   ETH = Ethereum
%   XRP = Ripple
%   LTC = Litecoin
%   XLM = Stellar.

% The data goes from August 7th, 2015 to November 4th, 2019.
%   /!\ IMPORT THE "returns.csv" FILE IN CURRENT FOLDER /!\
returns = table2array(readtable('returns.csv'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Table 1. Data on market capitalization, dominance and
%          portfolio allocation

% Market cap as of 5/11/2019
cap_BTC = 169058027086;
cap_ETH = 20641879383;
cap_XRP = 13015567545;
cap_LTC = 4039983342;
cap_XLM = 1647873393;
sum_cap = cap_BTC + cap_ETH + cap_XRP + cap_LTC + cap_XLM;

% Allocations
a = ([cap_BTC cap_ETH cap_XRP cap_LTC cap_XLM] / sum_cap)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Table 2. Summary statistics of returns

min(returns);
max(returns);
mean(returns);
median(returns);
quantile(returns, 0.25);
quantile(returns, 0.75);
std(returns);
skewness(returns);
kurtosis(returns);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Preliminary estimated quantities needed for the calculations

% mean returns
mu_hat = (mean(returns))';
% variance-covariance matrix of the portfolio
sigma_hat = cov(returns);
% portfolio losses (-portfolio returns)
z_t = -returns*a;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VaR estimation using the Gaussian approximation

% alpha=5%
VaR_Gauss_5 = -a' * mu_hat + (a' * sigma_hat * a)^(1/2) * ...
    norminv(0.95);
% alpha=1%
VaR_Gauss_1 = -a' * mu_hat + (a' * sigma_hat * a)^(1/2) * ...
    norminv(0.99);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VaR estimation using historical data

% alpha=5%
VaR_historical_5 = quantile(-returns*a, 0.95);
% alpha=1%
VaR_historical_1 = quantile(-returns*a, 0.99);

% Figure 1 : Empirirical loss distribution
figure;
histfit(z_t)
xlabel('Losses');
title('Historical data');
xline(VaR_historical_5, 'b', 'VaR (5%)');
xline(VaR_historical_1, 'r', 'VaR (1%)');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VaR estimation using gaussian kernel

% alpha = 5%
VaR_grid_5 = 0.04:0.001:0.1;
loss_estimate_5 = zeros([1 size(VaR_grid_5, 2)]);
for i = 1:size(VaR_grid_5, 2)
    loss_estimate_5(1, i) = loss_function(z_t, ...
        VaR_grid_5(i), 0.05);
end
[minimum, idx] = min(loss_estimate_5);
VaR_5_kernel = VaR_grid_5(idx);

% Figure 2. Loss (L) function at 5%
figure;
plot(VaR_grid_5, loss_estimate_5);
text(0.065, 0.0015, 'VaR=0.061 (5%)');
ylabel('Loss function');
xlabel('VaR');
xline(VaR_5_kernel, 'LineWidth', 1.2);

% alpha = 1%
VaR_grid_1 = 0.07:0.0005:0.14;
loss_estimate_1 = zeros([1 size(VaR_grid_1, 2)]);
for i = 1:size(VaR_grid_1, 2)
    loss_estimate_1(1, i) = loss_function(z_t, ...
        VaR_grid_1(i), 0.01);
end
[minimum_2, idx_2] = min(loss_estimate_1);
VaR_1_kernel = VaR_grid_1(idx_2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimating VaR using GPD

% Mean excess plot to find threshold u*
me  = sort(z_t);
u = sort(z_t);
for i = 1:(size(z_t, 1)-1)
    data = z_t(z_t > u(i));
    me(i) = mean(data - u(i));
end

% tentative search of u* -> visual approach
% Figure 3 : Mean excess plot
figure;
ha = area([0.01 0.07], [0.25 0.25]);
ha.FaceAlpha=0.1;
hold on
plot(u, me, 'o')
xlabel('u')
ylabel('Mean Excess')
xline(0.01, 'b', 'u* region');
xline(0.07, 'b');
hold off

% Preliminary calculations to estimate VaR using GPD
u_x = 0.05;
xi_hat = 0.5 * (1 - (((mean(z_t(z_t>u_x)) - u_x)^2) / ...
    var(z_t(z_t > u_x))));
sigma_hat_GPD = 0.5*(mean(z_t(z_t > u_x)) - u_x) * ...
    ((((mean(z_t(z_t > u_x)) - u_x)^2)/var(z_t(z_t > u_x)))+1);

sigma_tilde = sigma_hat_GPD * (size(u(u>u_x), 1) / ...
    size(z_t, 1))^(xi_hat);
mu_tilde = u_x - (sigma_tilde / xi_hat) * ...
    (((size(u(u>u_x), 1) / size(z_t, 1))^(-xi_hat)) - 1);

% alpha = 0.05
VaR_GPD_5 = mu_tilde + (sigma_tilde / xi_hat) * ...
    ((0.05^(-xi_hat)) - 1);

% alpha = 0.01
VaR_GPD_1 = mu_tilde + (sigma_tilde / xi_hat) * ...
    ((0.01^(-xi_hat)) - 1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ES estimation using the gaussian approximation

% alpha=5%
ES_Gauss_5 = -a' * mu_hat + (a' * sigma_hat * a)^(1/2) *...
    normpdf(norminv(0.95))/0.05;
% alpha=1%
ES_Gauss_1 = -a' * mu_hat + (a' * sigma_hat * a)^(1/2) *...
    normpdf(norminv(0.99))/0.01;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ES estimation using historical data

% alpha=5%
ES_historical_5 = mean(z_t(z_t > VaR_historical_5));
% alpha=1%
ES_historical_1 = mean(z_t(z_t > VaR_historical_1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ES estimation using gaussian kernel

% Estimation of the bandwidth
h = std(z_t) * size(returns, 1)^(-1/5);

% alpha = 5%
numerator_ES_5 = 0;
alpha_5 = 0.05;
for i = 1:size(returns, 1)
    numerator_ES_5 = numerator_ES_5 - a' * returns(i, :)' * ...
        normcdf((-a' * returns(i, :)'- VaR_5_kernel)/h); 
end
ES_kernel_5 = (numerator_ES_5 / size(returns, 1)) / alpha_5;

% alpha = 1%
numerator_ES_1 = 0;
alpha_1 = 0.01;
for i = 1:size(returns, 1)
    numerator_ES_1 = numerator_ES_1 - a' * returns(i, :)' * ...
        normcdf((-a' * returns(i, :)'- VaR_1_kernel)/h); 
end
ES_kernel_1 = (numerator_ES_1 / size(returns, 1)) / alpha_1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ES using GPD

% alpha = 5%
ES_GPD_5 = VaR_GPD_5 - (sigma_tilde / (xi_hat - 1)) * ...
    (0.05^(-xi_hat));

% alpha = 1%
ES_GPD_1 = VaR_GPD_1 - (sigma_tilde / (xi_hat - 1)) * ...
    (0.01^(-xi_hat));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation of confidence intervals using sub-sampling

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gaussian approximation

% alpha 5%
T = size(returns, 1);
b = ceil(0.5*T);
nboot = T - b + 1;
VaR_subsampling_Gauss_5 = zeros(nboot, 1);
ES_subsampling_Gauss_5 = zeros(nboot, 1);
for i = 1:nboot
    returns_star = returns(i:(i + b - 1), :);
    mu_hat = mean(returns_star);
    sigma_hat = cov(returns_star);
    VaR_subsampling_Gauss_5(i) = -a' * mu_hat' + ...
        (a' * sigma_hat * a)^(1/2) * norminv(0.95);
    ES_subsampling_Gauss_5(i) = -a' * mu_hat' + ...
        (a' * sigma_hat * a)^(1/2) *...
        normpdf(norminv(0.95))/0.05;
end

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for VaR
LB_VaR_subsampling_Gauss_5 = quantile(VaR_subsampling_Gauss_5, 0.025);
UB_VaR_subsampling_Gauss_5 = quantile(VaR_subsampling_Gauss_5, 0.975);

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for ES
LB_ES_subsampling_Gauss_5 = quantile(ES_subsampling_Gauss_5, 0.025);
UB_ES_subsampling_Gauss_5 = quantile(ES_subsampling_Gauss_5, 0.975);

% alpha 1%
T = size(returns, 1);
b = ceil(0.5*T);
nboot = T - b + 1;
VaR_subsampling_Gauss_1 = zeros(nboot, 1);
ES_subsampling_Gauss_1 = zeros(nboot, 1);
for i = 1:nboot
    returns_star = returns(i:(i + b - 1), :);
    mu_hat = mean(returns_star);
    sigma_hat = cov(returns_star);
    VaR_subsampling_Gauss_1(i) = -a' * mu_hat' + ...
        (a' * sigma_hat * a)^(1/2) * norminv(0.99);
    ES_subsampling_Gauss_1(i) = -a' * mu_hat' + ...
        (a' * sigma_hat * a)^(1/2) * normpdf(norminv(0.99))/0.01;
end

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for VaR
LB_VaR_subsampling_Gauss_1 = quantile(VaR_subsampling_Gauss_1, 0.025);
UB_VaR_subsampling_Gauss_1 = quantile(VaR_subsampling_Gauss_1, 0.975);

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for ES
LB_ES_subsampling_Gauss_1 = quantile(ES_subsampling_Gauss_1, 0.025);
UB_ES_subsampling_Gauss_1 = quantile(ES_subsampling_Gauss_1, 0.975);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Historical data

% alpha 5%
T = size(returns, 1);
b = ceil(0.5*T);
nboot = T - b + 1;
VaR_subsampling_historical_5 = zeros(nboot, 1);
ES_subsampling_historical_5 = zeros(nboot, 1);
for i = 1:nboot
    z_t_star = z_t(i:(i + b - 1));
    VaR_subsampling_historical_5(i) = quantile(z_t_star, 0.95);
    ES_subsampling_historical_5(i) = mean(z_t_star(z_t_star > ...
        VaR_subsampling_historical_5(i)));
end

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for VaR
LB_VaR_subsampling_5 = quantile(VaR_subsampling_historical_5, 0.025);
UB_VaR_subsampling_5 = quantile(VaR_subsampling_historical_5, 0.975);

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for ES
LB_ES_subsampling_5 = quantile(ES_subsampling_historical_5, 0.025);
UB_ES_subsampling_5 = quantile(ES_subsampling_historical_5, 0.975);

% alpha 1%
T = size(returns, 1);
b = ceil(0.5*T);
VaR_subsampling_historical_1 = zeros(nboot, 1);
ES_subsampling_historical_1 = zeros(nboot, 1);
for i = 1:nboot
    z_t_star = z_t(i:(i + b - 1));
    VaR_subsampling_historical_1(i) = quantile(z_t_star, 0.99);
    ES_subsampling_historical_1(i) = mean(z_t_star(z_t_star > ...
        VaR_subsampling_historical_1(i)));
end

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for VaR
LB_VaR_subsampling_historical_1 = quantile(VaR_subsampling_historical_1, 0.025);
UB_VaR_subsampling_historical_1 = quantile(VaR_subsampling_historical_1, 0.975);

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for ES
LB_ES_subsampling_1 = quantile(ES_subsampling_historical_1, 0.025);
UB_ES_subsampling_1 = quantile(ES_subsampling_historical_1, 0.975);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gaussian Kernel
% CAUTION : This section of the script can take several minutes
%           because there is a brute force search of the
%           minimum of the loss function over each subsample
% /!\ Uncomment to estimate the confidence intervals /!\

% % alpha = 5%
% T = size(returns, 1);
% b = ceil(0.5*T);
% nboot = T - b + 1;
% VaR_subsampling_kernel_5 = zeros(nboot, 1);
% ES_subsampling_kernel_5 = zeros(nboot, 1);
% for i = 1:nboot
%     z_t_star = z_t(i:(i + b - 1));
%     VaR_grid_5 = 0.04:0.002:0.1;
%     loss_estimate_5 = zeros([1 size(VaR_grid_5, 2)]);
%     for j = 1:size(VaR_grid_5, 2)
%         loss_estimate_5(1, j) = loss_function(z_t_star, ...
%             VaR_grid_5(j), 0.05);
%     end
%     [minimum, idx] = min(loss_estimate_5);
%     VaR_subsampling_kernel_5(i) = VaR_grid_5(idx);
%     % ES
%     numerator_ES_5 = 0;
%     alpha_5 = 0.05;
%     returns_star = returns(i:(i + b - 1), :);
%     h = std(z_t_star) * size(returns_star, 1)^(-1/5);
%     for k = 1:b
%         numerator_ES_5 = numerator_ES_5 - a' * returns_star(k, :)' * ...
%         normcdf((-a' * returns_star(k, :)'- VaR_subsampling_kernel_5(i))/h); 
%     end
%     ES_subsampling_kernel_5(i) = (numerator_ES_5 / size(returns_star, 1)) /...
%         alpha_5;
% end

% % Lower bound (LB) and Upper bound (UB) Conf. Inter. for VaR
% LB_VaR_subsampling_kernel_5 = quantile(VaR_subsampling_kernel_5, 0.025);
% UB_VaR_subsampling_kernel_5 = quantile(VaR_subsampling_kernel_5, 0.975);

% % Lower bound (LB) and Upper bound (UB) Conf. Inter. for ES
% LB_ES_subsampling_kernel_5 = quantile(ES_subsampling_kernel_5, 0.025);
% UB_ES_subsampling_kernel_5 = quantile(ES_subsampling_kernel_5, 0.975);

% % alpha = 1%
% T = size(returns, 1);
% b = ceil(0.5*T);
% nboot = T - b + 1;
% VaR_subsampling_kernel_1 = zeros(nboot, 1);
% ES_subsampling_kernel_1 = zeros(nboot, 1);
% for i = 1:nboot
%     z_t_star = z_t(i:(i + b - 1));
%     VaR_grid_1 = 0.08:0.002:0.12;
%     loss_estimate_1 = zeros([1 size(VaR_grid_1, 2)]);
%     for j = 1:size(VaR_grid_1, 2)
%         loss_estimate_1(1, j) = loss_function(z_t_star, VaR_grid_1(j), 0.01);
%     end
%     [minimum, idx] = min(loss_estimate_1);
%     VaR_subsampling_kernel_1(i) = VaR_grid_1(idx);
%     % ES
%     numerator_ES_1 = 0;
%     alpha_1 = 0.01;
%     returns_star = returns(i:(i + b - 1), :);
%     h = std(z_t_star) * size(returns_star, 1)^(-1/5);
%     for k = 1:b
%         numerator_ES_1 = numerator_ES_1 - a' * returns_star(k, :)' * ...
%         normcdf((-a' * returns_star(k, :)'- VaR_subsampling_kernel_1(i))/h); 
%     end
%     ES_subsampling_kernel_1(i) = (numerator_ES_1 / size(returns_star, 1)) /...
%         alpha_1;
% end

% % Lower bound (LB) and Upper bound (UB) Conf. Inter. for VaR
% LB_VaR_subsampling_kernel_1 = quantile(VaR_subsampling_kernel_1, 0.025);
% UB_VaR_subsampling_kernel_1 = quantile(VaR_subsampling_kernel_1, 0.975);

% % Lower bound (LB) and Upper bound (UB) Conf. Inter. for ES
% LB_ES_subsampling_kernel_1 = quantile(ES_subsampling_kernel_1, 0.025);
% UB_ES_subsampling_kernel_5 = quantile(ES_subsampling_kernel_1, 0.975);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimation using GPD and POT

% alpha 5%
T = size(returns, 1);
b = ceil(0.5*T);
nboot = T - b + 1;
VaR_subsampling_GPD_5 = zeros(nboot, 1);
VaR_subsampling_GPD_1 = zeros(nboot, 1);
ES_subsampling_GPD_5 = zeros(nboot, 1);
ES_subsampling_GPD_1 = zeros(nboot, 1);
u_x = 0.05;
for i = 1:nboot
    z_t_star = z_t(i:(i + b - 1));
    u = sort(z_t_star);
    xi_hat = 0.5 * (1 - (((mean(z_t_star(z_t_star>u_x)) - u_x)^2) /...
        var(z_t_star(z_t_star > u_x))));
    sigma_hat_GPD = 0.5*(mean(z_t_star(z_t_star > u_x)) - u_x) *...
        ((((mean(z_t_star(z_t_star > u_x)) - u_x)^2) / ...
        var(z_t_star(z_t_star > u_x)))+1);
    sigma_tilde = sigma_hat_GPD * (size(u(u>u_x), 1) / ...
        size(z_t_star, 1))^(xi_hat);
    mu_tilde = u_x - (sigma_tilde / xi_hat) * (((size(u(u>u_x), 1) / ...
        size(z_t_star, 1))^(-xi_hat)) - 1);
    % alpha = 0.05
    VaR_subsampling_GPD_5(i) = mu_tilde + (sigma_tilde / xi_hat) * ...
        ((0.05^(-xi_hat)) - 1);
    ES_subsampling_GPD_5(i) = VaR_subsampling_GPD_5(i) - ...
        (sigma_tilde / (xi_hat - 1))*(0.05^(-xi_hat));
    % alpha = 0.01
    VaR_subsampling_GPD_1(i) = mu_tilde + (sigma_tilde / xi_hat) * ...
        ((0.01^(-xi_hat)) - 1);
    ES_subsampling_GPD_1(i) = VaR_subsampling_GPD_1(i) - ...
        (sigma_tilde / (xi_hat - 1))*(0.01^(-xi_hat));
end

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for VaR
LB_VaR_subsampling_GPD_5 = quantile(VaR_subsampling_GPD_5, 0.025);
UB_VaR_subsampling_GPD_5 = quantile(VaR_subsampling_GPD_5, 0.975);
LB_VaR_subsampling_GPD_1 = quantile(VaR_subsampling_GPD_1, 0.025);
UB_VaR_subsampling_GPD_1 = quantile(VaR_subsampling_GPD_1, 0.975);

% Lower bound (LB) and Upper bound (UB) Conf. Inter. for ES
LB_ES_subsampling_GPD_5 = quantile(ES_subsampling_GPD_5, 0.025);
UB_ES_subsampling_GPD_5 = quantile(ES_subsampling_GPD_5, 0.975);
LB_ES_subsampling_GPD_1 = quantile(ES_subsampling_GPD_1, 0.025);
UB_ES_subsampling_GPD_1 = quantile(ES_subsampling_GPD_1, 0.975);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Figure 4. VaR and ES boxplots

figure;
boxplot([VaR_subsampling_Gauss_5, VaR_subsampling_historical_5,...
    VaR_subsampling_kernel_5, VaR_subsampling_GPD_5], ...
    'Labels', {'normal', 'historical', 'kernel', 'GPD'});
set(gca, 'XTickLabelRotation',90);
ylabel('VaR (5%)');

% figure;
boxplot([VaR_subsampling_Gauss_1, VaR_subsampling_historical_1,...
    VaR_subsampling_kernel_1, VaR_subsampling_GPD_1], ...
    'Labels', {'normal', 'historical', 'kernel', 'GPD'});
set(gca, 'XTickLabelRotation',90);
ylabel('VaR (1%)');

% figure;
boxplot([ES_subsampling_Gauss_5, ES_subsampling_historical_5, ES_subsampling_kernel_5, ES_subsampling_GPD_5], ...
    'Labels', {'normal', 'historical', 'kernel', 'GPD'});
set(gca, 'XTickLabelRotation',90);
ylabel('ES (5%)');

% figure;
boxplot([ES_subsampling_Gauss_1, ES_subsampling_historical_1, ES_subsampling_kernel_1, ES_subsampling_GPD_1], ...
    'Labels', {'normal', 'historical', 'kernel', 'GPD'});
set(gca, 'XTickLabelRotation',90);
ylabel('ES (1%)');

