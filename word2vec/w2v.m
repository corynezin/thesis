%% Word2Vec
clear
filename = "exampleWordEmbedding.vec";
emb = readWordEmbedding(filename);
girl = word2vec(emb,"girl");
man = word2vec(emb,"man");
woman = word2vec(emb,"woman");
word = vec2word(emb,girl-woman);

%% Saliency
clear
[x,t] = simplefit_dataset;
net = feedforwardnet(10);
net = train(net,x,t);
y = net(x);
perf = perform(net,t,y);
dwb = defaultderiv('dperf_dwb',net,x,t);
dfp = fpderiv('dperf_dwb',net,x,t);

%% LSTMs
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

