clear
clc

full_path='/storage/work/mjv5465/SAFE/';

restoredefaultpath;
addpath(genpath([full_path,'Data/']));
addpath(genpath([full_path,'Functions']));
addpath(genpath([full_path,'Scratch/']));
cd([full_path,'Scratch/']);

clear full_path 

%%

clear
clc

load dates
load me
load ret
load NYSE
load tcosts
load ff

% Read the signals data
[anoms23, labels23] = getAnomalySignals('novyMarxVelikovAnomalies.csv', 1, 2);
[nMonths, nStocks, nAnoms] = size(anoms23);

% Fill those in
filledAnoms = nan(nMonths, nStocks, nAnoms);
for i = 1:nAnoms
    % Fill the months for the annual ones
    var = FillMonths(anoms23(:, :, i));
    
    % Create a monthly rank and standardize to be between -0.5 and 0.5
    var = tiedrank(var')';
    var = (var - 1) ./ (max(var, [], 2)-1);
    var = var - 0.5;        
    
    % Fill the observations with market with the median for this
    % characteristic (i.e., 0)
    indToZero = isnan(var) & isfinite(me);    
    var(indToZero) = 0;
    
    % Set to NaN all observations where we don't have a market cap
    var(isnan(me)) = nan;
    
    % Assign to the new 3-d array
    filledAnoms(:, :, i) = var;      
end

% Store a few constants
T = 120;
s = find(dates==197306);
nObs = nMonths * nStocks;

% Permute the characteristics
chars = permute(filledAnoms, [2 1 3]);

% Prepare the matrices for the LASSO estimation
lassoY = reshape(ret, nObs, 1);
lassoX = [];

% Lag & reshape the characteristics
for i = 1:nAnoms
    temp = lag(chars(:, :, i)', 1, nan);
    lassoX = [lassoX reshape(temp, nObs, 1)];
end

% Reshape the dates too
lassoDates = reshape(repmat(dates, 1, nStocks), nObs, 1);

% Filter
ind = isfinite(sum([lassoY lassoX], 2));
lassoY = lassoY(ind, :);
lassoX = lassoX(ind, :);
lassoDates = lassoDates(ind, :);

% Store the annual variables
startYear = floor(dates(s+T)/100);
endYear = floor(dates(end)/100);
nYears = endYear - startYear + 1;

% Initialize the coefficient matrices
bhat_lasso = nan(nYears, nAnoms);
bhat_enet = nan(nYears, nAnoms);
bhat_ridge = nan(nYears, nAnoms);
bhat_olsh = nan(nYears, nAnoms);

%% We'll parallelize here & estimate coefficients once a year

parfor yr = 1:nYears 
    tic;

    % Find the month
    thisYear = startYear + yr - 1;
    i = find(dates==thisYear*100+12);

    % Timekeeping
    dates(i)
    
    % Find the past T months
    ind = lassoDates <= dates(i) & ...
          lassoDates >= dates(i-T+1);
    y = lassoY(ind);
    x = lassoX(ind,:);
    % x_t = permute(chars(:, i, :), [1 3 2]);

    % Estimate LASSO with 5-fold cross-validation
    [lassoB, lassoFitInfo] = lasso(x, y, 'CV', 5);
    bhat_lasso(yr, :) = lassoB(:, lassoFitInfo.IndexMinMSE)';

    % Estimate elastic net with 5-fold cross-validation (rho = 0.5)
    [enetB, enetFitInfo] = lasso(x, y, 'CV', 5, 'Alpha', 0.5);
    bhat_enet(yr, :) = enetB(:, enetFitInfo.IndexMinMSE)';

    % Estimate ridge regression with 5-fold cross-validation (rho -> 0)
    [ridgeB, ridgeFitInfo] = lasso(x, y, 'CV', 5, 'Alpha', 0.001);
    bhat_ridge(yr, :) = ridgeB(:, ridgeFitInfo.IndexMinMSE)';


    % OLS-Huber
    olsHuberB = robustfit(x, y, 'huber');
    bhat_olsh(yr, :) = olsHuberB(2:end)';
     
    % Random forest
    Mdl = fitrensemble(x,y);
    mdl_rf{yr, 1} = Mdl;
    
    % Neural network - 3 hidden layers
    tempNet = feedforwardnet([32 16 8]);
    % view(tempNet)
    % tempNet.layers{1}  % You can see the specifications of the first hidden layer
    tempNet.layers{1}.transferFcn = 'poslin'; % 'poslin' stands for positive linear - that's the ReLU activation function
    tempNet.layers{2}.transferFcn = 'poslin'; % 'poslin' stands for positive linear - that's the ReLU activation function
    tempNet.layers{3}.transferFcn = 'poslin'; % 'poslin' stands for positive linear - that's the ReLU activation function
    % view(tempNet)
    % tempNet.divideParam
    [tempNet, tr] = train(tempNet, x', y', 'showResources', 'no');
    mdl_nn3{yr, 1} = tempNet;
    toc
end

save Data/ml_bhat bhat_lasso bhat_enet bhat_ridge bhat_olsh mdl_nn3 

%% Calculate the expected returns

% Initialize the expected returns
eret_lasso = nan(nMonths, nStocks);
eret_enet = nan(nMonths, nStocks);
eret_ridge = nan(nMonths, nStocks);
eret_olsh = nan(nMonths, nStocks);
eret_nn3 = nan(nMonths, nStocks);

tic;
for yr = startYear:endYear-1
    
    % Find the index of the yr
    k = yr - startYear + 1;


    % Find the month
    i = find(dates == yr*100+12);
    
    % Timekeeping
    dates(i)
    
    for j = i:i+11

        % Find the x_t's
        x_t = permute(chars(:, j, :), [1 3 2]);
    
        % Estimate LASSO with 5-fold cross-validation
        eret_lasso(j, :) = (x_t * bhat_lasso(k,:)')';
        eret_enet(j, :) = (x_t * bhat_enet(k,:)')';
        eret_ridge(j, :) = (x_t * bhat_ridge(k,:)')';
        eret_olsh(j, :) = (x_t * bhat_olsh(k,:)')';
            
        % Neural network
        tempNet = mdl_nn3{k};
        eret_nn3(j, :) = sim(tempNet, x_t');


    end
end
toc

save Data/ml_eret eret_lasso eret_enet eret_ridge eret_olsh eret_nn3


%%

clear
clc

load ml_eret
load dates
load me
load ret
load NYSE
load ff

% Collect the expected returns in a structure
eret(1).var = eret_lasso;
eret(1).label = {'LASSO'};
eret(2).var = eret_enet;
eret(2).label = {'E-NET'};
eret(3).var = eret_ridge;
eret(3).label = {'Ridge'};
eret(4).var = eret_olsh;
eret(4).label = {'OLS-Huber'};
eret(5).var = eret_nn3;
eret(5).label = {'NN3'};

% Store the number of expected return variables we have
nErets = length(eret);

% initialize the portfolio returns matrix
prets = nan(length(dates), nErets);

for i = 1:nErets
    ind = makeUnivSortInd(eret(i).var, 5, NYSE);
    tempRes = runUnivSort(ret, ind, dates, me, 'printResults', 0, ...
                                               'plotFigure', 0);
    
    prets(:,i) = tempRes.pret(:,end);
end

% Look at performance over the entire period
ibbots([prets mkt], dates, 'legendLabels', [[eret.label],{'MKT'}], ...
                           'timePeriod', 198501);

% Start in 2005 to see difference
ibbots([prets mkt], dates, 'legendLabels', [[eret.label],{'MKT'}], ...
                           'timePeriod', 200501);
