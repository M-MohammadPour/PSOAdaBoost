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


% Test new Sample

newX=[-1 2];
sum=0;
for i = 1:iter
    weakClassifier=abClassifier.WeakClas{i};
    label = predStump(newX, weakClassifier);
    sum=sum+abClassifier.Weight(i)*label; % N x 1
end


N = size(X,1);
ma = {'ks','ko'};
fc = {'r','b'};
tv = unique(Y);
figure(2); hold off

for i = 1:length(tv)
    pos = find(Y==tv(i));
    plot(trnX(pos,1),trnX(pos,2),ma{i},'LineWidth',2,'MarkerSize',10, 'MarkerEdgeColor','black','MarkerFaceColor',fc{i});
    hold on
end

grid on
set(gca,'YDir','reverse');

hold on;
if(sign(sum)==-1)
    plot(newX(1,1),newX(1,2),'ks','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','black','MarkerFaceColor','g');
else
    plot(newX(1,1),newX(1,2),'ko','LineWidth',2,'MarkerSize',10,'MarkerEdgeColor','black','MarkerFaceColor','g');
end

