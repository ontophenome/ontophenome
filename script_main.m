%% output file
output_filepath = './results/output_LP.mat';

%% ontology to train (HPO / GO)
evaluation_ontology = 'HPO';

%% model to train (tlDLP / DLP / OGL / BiRW / LP)
model = 'LP';

%% PPI-Network (ML / BP)
gene_ontology = 'MF';

%% train model with all data
train_model = false;

if strcmp(model, 'OGL') || strcmp(model, 'DLP')
    %% OGL/DLP hyperparams grid
    mv_betas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6];
    mv_gammas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6];
    mv_Lengths = [length(mv_betas), length(mv_gammas)];
    params = zeros(prod(mv_Lengths), 2);
    % gamma
    params(:,2) = repmat(mv_gammas', [prod(mv_Lengths(1)), 1]);
    % beta
    mv_Vec = reshape(repmat(mv_betas, [prod(mv_Lengths(2)),1]), [prod(mv_Lengths(1:2)),1]);
    params(:,1) = mv_Vec;
elseif strcmp(model, 'tlDLP')
    %% tlDLP hyperparams grid
    mv_betas_X = 1e-4;
    mv_gammas_X = 1e-5;
    mv_betas_Y = 1e-4;
    mv_gammas_Y = 1e6;
    mv_zeta = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3];
    mv_Lengths = [length(mv_betas_X), length(mv_gammas_X), length(mv_betas_Y), length(mv_gammas_Y), length(mv_zeta)];
    params = zeros(prod(mv_Lengths), 5);
    % zeta
    params(:,5) = repmat(mv_zeta', [prod(mv_Lengths(1:4)), 1]);
    % gamma Y
    mv_Vec = reshape(repmat(mv_gammas_Y,[mv_Lengths(5),1]), [prod(mv_Lengths(4:5)),1]);
    params(:,4) = repmat(mv_Vec, [prod(mv_Lengths(1:3)), 1]);
    % beta Y
    mv_Vec = reshape(repmat(mv_betas_Y,[mv_Lengths(4),1]), [prod(mv_Lengths(3:4)),1]);
    params(:,3) = repmat(mv_Vec, [prod(mv_Lengths(1:2)), 1]);
    % gamma X
    mv_Vec = reshape(repmat(mv_gammas_X,[mv_Lengths(3),1]), [prod(mv_Lengths(2:3)),1]);
    params(:,2) = repmat(mv_Vec, [prod(mv_Lengths(1:1)), 1]);
    % beta X
    mv_Vec = reshape(repmat(mv_betas_X, [prod(mv_Lengths(1:1)),1]), [prod(mv_Lengths(1:2)),1]);
    params(:,1) = mv_Vec;
elseif strcmp(model, 'LP')
    %% LP hyperparams grid
    params = 0.1:0.1:0.9;
elseif strcmp(model, 'BiRW')
    %% BiRW hyperparams grid
    m_vParamVecAlphas = 0.1:0.1:0.9;
    m_vParamVecLR = 1:4;
    mv_Lengths = [length(m_vParamVecAlphas), length(m_vParamVecLR)];
    params = zeros(prod(mv_Lengths), 2);
    % left/right
    params(:,2) = repmat(m_vParamVecLR', [prod(mv_Lengths(1)), 1]);
    % alpha
    mv_Vec = reshape(repmat(m_vParamVecAlphas, [prod(mv_Lengths(2)),1]), [prod(mv_Lengths(1:2)),1]);
    params(:,1) = mv_Vec;
end

%% data path
go_filepath = './data/gene_function_data/human/processed_data/MF_data.mat';
po_filepath = './data/phenotype_data/human/processed_data_ppi_version/all_data.mat';
ppi_filepath = './data/ppi_data/distance/ppi_MF.mat';
cv_index_filepath = './data/CVindex/elementwise/HPO_CVset_all_5foldCV.mat';

% tlDLP initialization file
% if running tlDLP, need to provide Y0, usually output from DLP or OGL
tdlp_Y0 = load('./data/DLP_Y_hat.mat');
tdlp_Y0 = tdlp_Y0.Yhat;

%% CV folds
fold_start = 1;
fold_end = 5;

%% call model
runModel(evaluation_ontology, ...
            model, ...
            params, ...
            gene_ontology, ...
            output_filepath, ...
            fold_start, ...
            fold_end, ...
            train_model, ...
            go_filepath, ...
            po_filepath, ...
            ppi_filepath, ...
            cv_index_filepath, ...
            tdlp_Y0);
