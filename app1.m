close all; clear; clc;

% Create Random Samples
N1 = 200;
N2 = 100;
X = [randn(N1,2)+2;randn(N2,2)+3];
Y = [zeros(N1,1);ones(N2,1)];
Y(Y==0) = -1;


N=size(X,1);
trnX = X(1:N, :);
trnY = Y(1:N);


iter = 20;
abClassifier = initAdaBoost(iter);

N = size(trnX, 1); % Number of training samples
sampleWeight = repmat(1/N, N, 1);

for t = 1:iter
    weakClassifier = buildStump(trnX, trnY, sampleWeight);

    
    abClassifier.WeakClas{t} = weakClassifier;
    abClassifier.nWC = t;
    % Compute the weight of this classifier
    abClassifier.Weight(t) = 0.5*log((1-weakClassifier.error)/weakClassifier.error);
    weakClassifier.error
    % Update sample weight
    label = predStump(trnX, weakClassifier);
    tmpSampleWeight = -1*abClassifier.Weight(t)*(trnY.*label); % N x 1
    tmpSampleWeight = sampleWeight.*exp(tmpSampleWeight); % N x 1
    
    sampleWeight = tmpSampleWeight./sum(tmpSampleWeight); % Normalized
    
    % Predict on training data
    [ttt, abClassifier.trnErr(t)] = predAdaBoost(abClassifier, trnX, trnY);
    
    
    fprintf('\tIteration %d, Training error %f\n', t, abClassifier.trnErr(t));
end


%     Plot Training Error
figure(1)
trnError = abClassifier.trnErr;
plot(1:iter, trnError); title('Trainin Error');


