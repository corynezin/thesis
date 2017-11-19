%% Load word embedding
filename = "exampleWordEmbedding.vec";
emb = readWordEmbedding(filename);
%% Loading training data
pos_dir = '/home/bcv/Documents/thesis/aclImdb/train/pos/';
neg_dir = '/home/bcv/Documents/thesis/aclImdb/train/neg/';
dp = dir(pos_dir);
dn = dir(neg_dir);

pos_files = {dp(~[dp(:).isdir]).name};
neg_files = {dn(~[dn(:).isdir]).name};
all_files = { pos_files{:}, neg_files{:} };
all_label = categorical(...
    [zeros(length(pos_files),1); ones(length(neg_files),1)]);
% Randomizing:
ind = randperm(length(pos_files)+length(neg_files));
all_label = all_label(ind);
all_files = all_files(ind);
%% 