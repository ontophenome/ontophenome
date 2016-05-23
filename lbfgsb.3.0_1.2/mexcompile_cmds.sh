cd Lbfgsb.3.0 

x86_64-w64-mingw32-gfortran.exe -c -Wall -Wno-uninitialized lbfgsb.f linpack.f blas.f timer.f 
echo EXPORTS > liblbfgsb.def 
echo setulb_ >> liblbfgsb.def 
echo _setulb_=setulb_ >> liblbfgsb.def

dlltool -D liblbfgsb.dll -d liblbfgsb.def -e liblbfgsb.o -l liblbfgsb.dll.a lbfgsb.o linpack.o blas.o timer.o 
lib /def:liblbfgsb.def /machine:x64

x86_64-w64-mingw32-gfortran.exe -mdll -static-libgfortran -o liblbfgsb.dll liblbfgsb.o lbfgsb.o linpack.o timer.o blas.o

cp liblbfgsb.dll liblbfgsb.dll.a liblbfgsb.lib . 
cp liblbfgsb.dll liblbfgsb.dll.a liblbfgsb.lib .. 
cp /usr/x86_64-w64-mingw32/sys-root/mingw/bin/libgcc_s_seh-1.dll .. 
cd ..

/cygdrive/c/Progra~1/MATLAB/R2013b/bin/mex.bat -O -largeArrayDims -UDEBUG lbfgsb_wrapper.c liblbfgsb.lib -lmex -lmx -lopenblas