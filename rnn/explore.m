% Name: Cory Nezin
% Date: 11/11/2017
% Goal: Unwrap a recurrent neural net
%% Initialize and train a neural net
clear
load JapaneseVowelsTrain
figure
for i = 1:12
    subplot(12,1,13-i)
    plot(X{1}(i,:));
    ylabel(i) 
    xticklabels('')
    yticklabels('')
    box off
end
title("Training Observation 1")
subplot(12,1,12)
xticklabels('auto')
xlabel("Time Step")

numObservations = numel(X);
for i=1:numObservations
    sequence = X{i};
    sequenceLengths(i) = size(sequence,2);
end

[sequenceLengths,idx] = sort(sequenceLengths);
X = X(idx);
Y = Y(idx);

miniBatchSize = 27;
miniBatchLocations = miniBatchSize+1:miniBatchSize:numObservations;
XLocations = repmat(miniBatchLocations,[2 1]);
YLocations = repmat([0;30],[1 9]);

inputSize = 12;
outputSize = 100;
outputMode = 'last';
numClasses = 9;

layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(outputSize,'OutputMode',outputMode)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer]

maxEpochs = 150;
miniBatchSize = 27;
shuffle = 'never';

options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle', shuffle);

net = trainNetwork(X,Y,layers,options);
%% Verify parameters
sprintf('Input is size %d',net.Layers(1).InputSize)
sprintf('LSTM layer takes %d inputs has %d outputs',...
    net.Layers(2).InputSize,net.Layers(2).OutputSize)
sprintf('FC layer takes %d inputs has %d outputs',...
    net.Layers(3).InputSize,net.Layers(3).OutputSize)
%% First Iteration:
rng(1)
x = randn(12,1);
h = zeros(100,1);
s = zeros(100,1);
N = 100;
A = net.Layers(2).InputWeights;
B = net.Layers(2).RecurrentWeights;
b = net.Layers(2).Bias;

sig = @(x) 1./(1+exp(-x));
t = @(x) tanh(x);
% How matlab updates the state and output:
%% What is really happening (defining operations):
% input gate
i = @(x,h) sig(A(1:N,:)*x + B(1:N,:)*h + b(1:N));
% forget gate
f = @(x,h) sig(A(N+1:2*N,:)*x + B(N+1:2*N,:)*h + b(N+1:2*N));
% layer input (input node)
g = @(x,h) t(A(2*N+1:3*N,:)*x + B(2*N+1:3*N,:)*h + b(2*N+1:3*N));
% output gate
o = @(x,h) sig(A(3*N+1:4*N,:)*x + B(3*N+1:4*N,:)*h + b(3*N+1:4*N));
%% The actual algorithm:
s = g(x,h).*i(x,h) + s.*f(x,h);
h = t(s).*o(x,h);
netup = classifyAndUpdateState(net,x);
% Confirming the MATLAB model is matched
actual_state = netup.Layers(2).CellState;
actual_outpt = netup.Layers(2).OutputState;
sum(abs(actual_state - s))
sum(abs(actual_outpt - h))
%% Next iteration:
s = g(x,h).*i(x,h) + s.*f(x,h);
h = t(s).*o(x,h);
netup = classifyAndUpdateState(netup,x);
actual_state = netup.Layers(2).CellState;
actual_outpt = netup.Layers(2).OutputState;
sum(abs(actual_state-(s)))
sum(abs(actual_outpt-(h)))

