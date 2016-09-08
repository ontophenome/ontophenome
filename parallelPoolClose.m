function parallelPoolClose()
    if exist('OCTAVE_VERSION', 'builtin') ~= 0
        return;
    else
        if exist('parpool') == 2
            poolobj = gcp('nocreate');
            delete(poolobj);
        elseif exist('matlabpool')
            matlabpool close force local;
        end
    end
end

