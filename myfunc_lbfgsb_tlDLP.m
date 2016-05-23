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
    
    VERBOSE = m_copt.verbose;
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
    wa      = ones(2*m*n + 5*n + 11*m*m + 8*m,1);
    iwa     = ones(3*n,1,'int32');
    task    = 'START';
    iprint  = 0;
    csave   = '';
    lsave   = zeros(4,1);
    isave   = zeros(44,1, 'int32');
    dsave   = zeros(29,1);
    f       = 0;
    g       = zeros(n,1);

    m_vobjs = zeros(maxIts, 1);
    %--- end of Proc()

    x   = reshape(W, [n, 1]) + 0; % important: we want Matlab to make a copy of this.
    %  'x' will be modified in-place

    outer_count     = 0;
    for k = 1:maxTotalIts
        % Call the mex file. The way it works is that you call it,
        %   then it returns a "task". If that task starts with 'FG',
        %   it means it is requesting you to compute the function and gradient,
        %   and then call the function again.
        % If it is 'NEW_X', it means it has completed one full iteration.
        [f, task, csave, lsave, isave, dsave] = ...
            lbfgsb_wrapper(m, x, l, u, nbd, f, g, factr, pgtol, wa, iwa, task, ...
                           iprint, csave, lsave, isave, dsave );

        W_full = reshape(x, [mn_J, mn_K]);
        W = sparse(W_full);
        
        task = deblank(task(1:60)); % this is critical!
        %otherwise, fortran interprets the string incorrectly

        if 1 == strfind( task, 'FG' )
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
            g(:) = reshape(Delta,[n, 1]);

        elseif 1 == strfind( task, 'NEW_X' )
            outer_count = outer_count + 1;

            % Display information if requested
            if ~mod( outer_count, printEvery ) && (VERBOSE > 0),
                m_vobjs(outer_count) = f;
                fprintf('Iteration %4d, f = %5.5e, ||g||_inf = %5.2e \n', ...
                    outer_count, f, norm(g,Inf) );
            end

            if outer_count >= maxIts
                disp('Maxed-out iteration counter, exiting...');
                break;
            end

        else
            break;
        end
    end

    m_vobjs = m_vobjs(1:outer_count);
end % end of main function