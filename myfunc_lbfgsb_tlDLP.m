function [W, m_vobjs] = myfunc_lbfgsb_tlDLP(W, W0, Z0, Btr, L, G2, m_copt)
    %--- Proc: Setting()
    [mn_J, mn_K] = size(W0);
    gamma = m_copt.gamma;

    gBtr =  gamma*Btr;
    BZ =  gamma*(Btr*Z0');
    
    Delta = zeros(mn_J, mn_K);

    % a new variable

    m_mWNzIDX = W0 > 0;
    WsubMat = zeros(sum(sum(m_mWNzIDX)), 1);
    
    WB = zeros(size(Z0));
    %--- end of Proc()


    %--- Proc: Options for L-BFGS-B()
    n = mn_J * mn_K;

    % Number of L-BFGS memory vectors
    m   = m_copt.m;

    l = zeros(n,1);
    u = inf*ones(n,1);
    nbd     = isfinite(l) + isfinite(u) + 2*isinf(l).*isfinite(u);
    factr   = m_copt.factor;
    pgtol   = m_copt.pgtol;

    maxIts  = m_copt.maxIts;
    maxTotalIts     = m_copt.maxTotalIts;
    printEvery = 1;

    % Make the work arrays
    iprint  = 0;
    %--- end of Proc()

    x   = reshape(W, [n, 1]) + 0; % important: we want Matlab to make a copy of this.
    %  'x' will be modified in-place

    callF = @(x) tlDLP_obj(x);
    fcn_wrapper(callF, printEvery, x);
    callF_wrapped = @(x,varargin) fcn_wrapper(callF, printEvery, x, varargin{:});
    
    [~,x,~,~, ~] = lbfgsb_wrapper( m, x, l, u, nbd, callF_wrapped, factr, pgtol, iprint, maxIts, maxTotalIts);
    m_vobjs = fcn_wrapper();
    W = reshape(x, [mn_J, mn_K]);
    
    function [f,g] = fcn_wrapper(callF, printEvery, x, varargin)
        persistent k history
        if isempty(k), k = 1; end
        if nargin==0
            % reset persistent variables and return information
            if ~isempty(history) && ~isempty(k) 
                printFcn(k,history);
                f = history(1:k,1);
            end
            history = [];
            k = [];
            return;
        end
        
        [f,g] = callF(x);
        
        if nargin > 3
            outerIter = varargin{1}+1;
            
            history(outerIter,1)    = f;
            history(outerIter,2)    = norm(g,Inf); % g is not projected
            
            if outerIter > k
                % Display info from *previous* input
                % Since this may be called several times before outerIter
                % is actually updated
                if ~isinf(printEvery) && ~mod(k,printEvery)
                    printFcn(k,history);
                end
                k = outerIter;
            end
            
        end

    end
    
    function [f,g] = tlDLP_obj(x) 
        W_full = reshape(x, [mn_J, mn_K]);
        W = sparse(W_full);

        % L-BFGS-B requests that we compute the gradient and function value
        % [f, g]

        % Delta = m_mMat01V + d{ 0.5*||W-W0||_W^2 + 0.5*gamma*||Z - W'SB'||_F^2 + 0.5*alpha*tr(W'LW) }/dW            
        WsubMat(:) = W_full(m_mWNzIDX) - W0(m_mWNzIDX);
        LW = sparse(L*W);
        GW = sparse(G2*W');
        WB(:,:) = W'*Btr;

        %- old: Delta(:,:) = gBtr*WB' - BZ + LW  + (W_full.*(m_mMat02V*G(m_vinnerN,:)));
        %- note: m_mMat02V*G(m_vinnerN,:) and m_mMat02V*sparse(G(m_vinnerN,:))
        %        do not yield the exactly same result. However, the
        %        differece would be negligible.
        Delta(:,:) = gBtr*WB' - BZ + LW + GW';
        Delta(m_mWNzIDX)  = Delta(m_mWNzIDX) + WsubMat;

        f = 0.5*sum(sum(W.*LW)) + 0.5*sum(sum(W'.*GW)) ...
            + 0.5*sum( WsubMat.^2 ) ...
            + 0.5*gamma*sum((Z0(:)-WB(:)).^2 );
        g = zeros(n,1);
        g(:) = reshape(Delta,[n, 1]);
    end
    
    function printFcn(k,history)
        fprintf('Iter %5d, f(x) = %2e, ||grad||_infty = %.2e', ...
            k, history(k,1), history(k,2) );
        fprintf('\n');
    end

end % end of main function