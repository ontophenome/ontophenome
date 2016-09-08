function parallelPoolStart()
    try
        if exist('OCTAVE_VERSION', 'builtin') ~= 0
            return;
        else
            if exist('parpool') == 2
                parpool;
            elseif exist('matlabpool')
                matlabpool;
            end
        end
    catch
    end
end

