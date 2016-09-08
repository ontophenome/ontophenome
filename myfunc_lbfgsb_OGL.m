function [W, m_vobjs] = myfunc_lbfgsb_OGL(W, W0, L, G, r, m_copt)
    %--- Proc: Setting()
    [mn_J, mn_K] = size(W0);

    mu = m_copt.mu;
    beta = m_copt.regul_glasso;

    rt = 0.5*beta*r;

    LW = zeros(mn_J, mn_K);
    Delta = zeros(mn_J, mn_K);
    
    m_vleafN = sum(G,2) == 1;
    m_vinnerN = ~m_vleafN;

    rt_inner = rt(m_vinnerN);
    rt_leaf = rt(m_vleafN);

    m_mMat01V = zeros(mn_J, sum(m_vinnerN));
    m_mMat02V = zeros(mn_J, sum(m_vinnerN));
    m_mMat01L = zeros(mn_J, sum(m_vinnerN)) > 0;

    Wds = zeros(mn_J, mn_K);
    m_mWNzIDX = W0 > 0;
    WsubMat = zeros(sum(sum(m_mWNzIDX)), 1);
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
    
    callF = @(x) OGL_obj(x);
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
    
    function [f,g] = OGL_obj(x) 
        W = reshape(x, [mn_J, mn_K]);
        Wds(:,:) = W.^2;
        
        m_mMat01V(:,:) = sqrt(Wds*G(m_vinnerN,:)');
        m_mMat01L(:,:) = m_mMat01V > repmat(mu./rt_inner,[mn_J,1]);

        m_mMat02V(:,:) = repmat(rt_inner/mu,[mn_J,1]);
        m_mMat02V(m_mMat01L) = 1./m_mMat01V(m_mMat01L);
        m_mMat02V(:,:) = m_mMat02V.*repmat(rt_inner,[mn_J,1]);

        m_mMat01V(:,:) = m_mMat01V.*repmat(rt_inner,[mn_J,1]);

        m_robjFt = sum(sum(m_mMat01V(m_mMat01L))) - 0.5*mu*sum(sum(m_mMat01L))...
            + sum(sum(m_mMat01V(~m_mMat01L).^2))/(2*mu);

        WsubMat(:) = W(m_mWNzIDX) - W0(m_mWNzIDX);
        LW(:,:) = (L*W);

        Delta(:,:) = LW + (W.*(m_mMat02V*G(m_vinnerN,:)));
        Delta(m_mWNzIDX)  = Delta(m_mWNzIDX) + WsubMat;
        Delta(:,m_vleafN) = Delta(:,m_vleafN) + repmat(rt_leaf,[mn_J,1]);

        f = m_robjFt ...
            + 0.5*sum(sum(W.*LW)) ...
            + sum( sum(W(:,m_vleafN),1).*rt_leaf )...
            + 0.5*sum( WsubMat.^2 );
        g = zeros(n,1);
        g(:) = reshape(Delta,[n, 1]);
        
    end

    function printFcn(k,history)
        fprintf('Iter %5d, f(x) = %2e, ||grad||_infty = %.2e', ...
            k, history(k,1), history(k,2) );
        fprintf('\n');
    end

end % end of main function