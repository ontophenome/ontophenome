/*=================================================================
* Accelerated gradient descent method + smooth approximation + line search
W  := X' and B:= Y (or W := Y' and B:= X)
rt := 0.5*beta*r (weights for each group)
m  := size(W,1), the number of the genes
||A||_M^2 := sum_i sum_j \pi(M_ij>0) A_ij^2

Instead of solving the orginal optimization problem,
>> F(W) = 0.5*||W-W0||_M^2 + 0.5*gamma*||Z - W'SB'||_F^2 + 0.5*alpha*tr(W'LW)
+ \sum_{i=1}^m\sum_{g\in G} rt(g) ||W_g||_2
= f(W) + g(W),
subject to W >= 0,
where
f(W) = 0.5*||W-W0||_M^2 + 0.5*gamma*||Z - W'SB'||_F^2 + 0.5*alpha*tr(W'LW)
+ \sum_{i=1}^m\sum_{g \in Inner nodes} rt(g) ||W_g||_2,
g(W) = \sum_{i=1}^m\sum_{g \in leaf nodes} rt(g) |W_g|,

we solve the Nesterov smooth approximation of F(W)
F_mu(W) := f_mu(W) + g(W)
f_mu(W) = 0.5*||W-W0||_M^2 + 0.5*gamma*||Z - W'SB'||_F^2 + 0.5*alpha*tr(W'LW)
+ max<CW',A> - mu*d(A)

programed by SPark.
References
1. "Tree-guided group lasso for multi-response regression
with structured sparsity, with an application to eQTL mapping",
Seyoung Kim and Eric P. Xing, Ann. Appl. Stat. 2012
2. "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems"
Amir Beck and Marc Teboulle, SIAM J. IMAGING SCIENCES, 2009

* solved by Limited memory Broyden�Fletcher�Goldfarb�Shannoy with box constraints
*
* Inputs (in order):
*  x       initial guess or current point, also used to determine the size of the problem (Nx1)

* Outputs:
* Hist
*
*
*
*=================================================================*/
#include <math.h>
#include "mex.h"

#include <string.h>
#include <limits.h> /* for CHAR_BIT */
#include <assert.h>

#ifdef _BLAS64_
#define int_F ptrdiff_t
#else
#define int_F int
#endif

/* these depend on the lbfgsb release version */
#define LENGTH_STRING 60
#define LENGTH_LSAVE 4
#define LENGTH_ISAVE 44
#define LENGTH_DSAVE 29

/* the fortran lbfgsb code uses lsave as type LOGICAL
* which is usually 'int' in C, whereas type LOGICAL*1
* would be 'bool' in C.
* BTW, the 'mxLogical' is usually defined as 'bool' */
typedef int fortranLogical;

#if MX_API_VER < 0x07030000
typedef int mwIndex;
#endif
#define Malloc(type,n) (type *)malloc((n)*sizeof(type));

#ifdef DEBUG
#define debugPrintf mexPrintf
#else
#define debugPrintf fakePrintf
#endif

/* For using fortran programs in C */
/* On Windows, we generally do not need to append the underscore to the end,
* whereas on Linux we do.
* Also, if using a native Windows fortran compiler, we may need
* to put the function names in all capital letters
* */
#if defined(NOUNDERSCORE)
#if defined(UPPERCASE_FORTRAN)
#define setulb SETULB
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define setulb SETULB_
#else
#define setulb setulb_
#endif
#endif

/* If the 'DEBUG' symbol is undefined, then don't print: */
int fakePrintf(const char *format, ...){
	return 0;
}


struct feature_node {
	int index;
	double value;
};

/* Declare the L-BFGS-B function */
void setulb(int_F *n, int_F *m, double *x, double *l, double *u,
	int_F *nbd, double *f, double *g, double *factr,
	double *pgtol, double *wa, int_F *iwa,
	char *task, int_F *iprint, char *csave, fortranLogical *lsave,
	int_F *isave, double *dsave);

/* This is taken from the other l-bfgs-b mex interface by Peter Carbonetto. */
/* Copy a C-style string (a null-terminated character array) to a
* non-C-style string (a simple character array). The length of the
* destination character array is given by "ndest". If the source is
* shorter than the destination, the destination is padded with blank
* characters.
* */
void copyCStrToCharArray(const char* source, char* dest, int ndest) {
	int i;
	int nsource = strlen(source);
	/* Only perform the copy if the source can fit into the destination. */
	if (nsource < ndest) {
		strcpy(dest, source);

		/* Fill in the rest of the string with blanks. */
		for (i = nsource; i < ndest; i++)
			dest[i] = ' ';
	}
}

mxLogical isInt(const mxArray *pm) {
	/* What size 'int' does the fortran program
	* expect ? Not sure... But let's hope that
	* it's the same size as the "int" in C.
	* On my 64-bit computer, CHAR_BIT = 8
	* and sizeof(int) = 4, so it's still 32 bits
	*
	* CHAR_BIT is from limits.h
	* If using gcc, you can run `gcc -dM -E - < /dev/null | grep CHAR_BIT`
	*  and it should define the symbol __CHAR_BIT__, so this is another way.
	* */

	/* debugPrintf("Sizeof(int) is %d\n", sizeof(int) ); */
	switch (CHAR_BIT * sizeof(int)) {
	case 16:
		return mxIsInt16(pm);
	case 32:
		return mxIsInt32(pm);
	case 64:
		return mxIsInt64(pm);
	default:
		mexErrMsgTxt("You have a weird computer that I don't know how to support");
		return false;
	}
}


/* Main mex gateway routine */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])   {
	int_F   iprint = 1;
	char    task[LENGTH_STRING], csave[LENGTH_STRING];
	int_F   *isave_int;
	mxLogical  *lsave_bool;
	fortranLogical    lsave[LENGTH_LSAVE];
	int_F   len_wdb, m_Hess, *nbd, *iwa, isave[LENGTH_ISAVE];
	double  *nbd_dbl, *iwa_dbl;
	double  f, factr, pgtol, *x, *l, *u, *g, dsave[LENGTH_DSAVE], *wa;
	mxClassID classID;

	int ndim = 2; /* for lcc compiler, must declare these here, not later ... */
	mwSize dims[2] = { LENGTH_ISAVE, 1 };

	int maxIts, maxTotalIts, outer_count, verbose;
	double mu, gamma;

	int iter, i, j, k, idx1, low, high, elements, mn_Cnt1, NposOpt, NposG;
	int mn_J, mn_K, mn_Ksub, mn_KL1, mn_K2, len_NZ;
	double mr_obj1, mr_obj2, mr_sum1, mr_sum2, mr_tmp1, mr_weight;
	double *mptr_Mat1, *mptr_Mat2, *mptr_Mat3, *mptr_Mat4, *mptr_Mat5;

	double *Wrow, *Wcol, *Wds, *W0, *rinner, *rleafE, *Z, *SBZ; /* *SBtr, *SBdb;*/
	uint64_T *NzIDX, *L1IDX;

	double *m_mGzeros, *mm_Weight;
	double *objHist;

	mwIndex *ir, *jc;
	mxArray *Grhs[1], *Glhs[1], *Wrhs[1], *Wlhs[1];

	struct feature_node *xi, *xi_Ref1, *xi_Ref2;

	struct feature_node **Gcol_idx;
	struct feature_node *Gcol_space;

	struct feature_node **Grow_idx;
	struct feature_node *Grow_space;

	struct feature_node **L_idx;
	struct feature_node *L_space;

	struct feature_node **SBtr_idx;
	struct feature_node *SBtr_space;

	/*
	struct feature_node **SBZ_idx;
	struct feature_node *SBZ_space;
	*/

	struct feature_node **SBdb_idx;
	struct feature_node *SBdb_space;


	if (nrhs < 13) {
		mexErrMsgTxt("The number of arguments should be 13.");
		mexErrMsgTxt("func(W,W0,NzIDX,G,rinner,rleaft,L1IDX,L,Z,SBtr,SBZ,SBdb,opt)");
	}
     
	/* Wrow = mxGetPr(prhs[0]); */
	W0 = mxGetPr(prhs[1]);
	NzIDX = (uint64_T *)mxGetPr(prhs[2]);
	/* Gsub = mxGetPr(prhs[3]); */
	rinner = mxGetPr(prhs[4]);
	rleafE = mxGetPr(prhs[5]);
	L1IDX = (uint64_T *)mxGetPr(prhs[6]);
	/*    Lmat   = mxGetPr(prhs[7]); */
	Z = mxGetPr(prhs[8]);
	/* SBtr = mxGetPr(prhs[9]); */
	SBZ = mxGetPr(prhs[10]);
	/* SBdb = mxGetPr(prhs[11]); */

	mn_K = (int)mxGetM(prhs[0]); /* mxGetM: number of rows */
	mn_J = (int)mxGetN(prhs[0]); /* mxGetN: number of columns */

	mn_Ksub = (int)mxGetM(prhs[3]); /* (Gsub) mxGetM: number of rows */

	mn_KL1 = mn_K - mn_Ksub;

	mn_K2 = (int)mxGetN(prhs[9]); /* (SBtr) mxGetN: number of columns */

	len_wdb = mn_K * mn_J;
	len_NZ = (int)mxGetM(prhs[2]); /* (NzIDX) mxGetM: number of rows */

	Wrhs[0] = mxDuplicateArray(prhs[0]);

    Wrow = Malloc(double, len_wdb);         
	Wds = Malloc(double, len_wdb);
	m_mGzeros = (double *)calloc(len_wdb, sizeof(double));
	mm_Weight = Malloc(double, mn_J*mn_Ksub);

	/* non-negativity contraints
	*  l       list of upper bounds (Nx1)
	*  u       list of lower bounds (Nx1)
	*  nbd     list of which bounds are active (Nx1):
	*          0=neither u nor l, 1 = only l, 2 = both u and l, 3 = only u
	*/
	nbd = (int_F *)mxMalloc(len_wdb * sizeof(int));
	l = Malloc(double, len_wdb);
	u = Malloc(double, len_wdb);

	memcpy(l, m_mGzeros, len_wdb*sizeof(double));
	for (i = 0; i<len_wdb; i++){
		nbd[i] = (int)1;
		u[i] = INFINITY;
	}
    

	/*  factr   stopping crit: 1e+12 for low accuracy, 1e7 for moderate, 1e1 for high accuracy
	*              (will be multiplied by machine precision)
	*  pgtol   stopping crit for infinity norm of projected gradient
	*/
	NposOpt = 12;

	gamma = (double)mxGetScalar(mxGetField(prhs[NposOpt], 0, "regul_glasso"));

	mu = (double)mxGetScalar(mxGetField(prhs[NposOpt], 0, "mu"));

	factr = (double)mxGetScalar(mxGetField(prhs[NposOpt], 0, "factor"));
	pgtol = (double)mxGetScalar(mxGetField(prhs[NposOpt], 0, "pgtol"));

	m_Hess = (int)mxGetScalar(mxGetField(prhs[NposOpt], 0, "m"));

	maxIts = (int)mxGetScalar(mxGetField(prhs[NposOpt], 0, "maxIts"));
	maxTotalIts = (int)mxGetScalar(mxGetField(prhs[NposOpt], 0, "maxTotalIts"));

	verbose = (int)mxGetScalar(mxGetField(prhs[NposOpt], 0, "verbose"));

	objHist = Malloc(double, maxIts);

	iprint = 1;

	/* the work arrays 'wa' and 'iwa'
	*  wa      work space array (double)
	*  iwa     work space array (int)
	*  wa      = (double *)mxMalloc( (2*m*n + 5*n + 11*m*m + 8*m ) * sizeof(double) );
	*  iwa     = (int_F *)mxMalloc( (3*n)*sizeof(int) );
	*/
	wa = (double *)mxMalloc((2 *m_Hess*len_wdb + 5*len_wdb + 11*m_Hess*m_Hess + 8*m_Hess) * sizeof(double));
	iwa = (int_F *)mxMalloc((3 *len_wdb)*sizeof(int));

	/* the 'task' string */
	copyCStrToCharArray("START", task, LENGTH_STRING);

	/* === Deal with the csave, lsave, isave and dsave variables === */
	copyCStrToCharArray("", csave, LENGTH_STRING);

	for (i = 0; i<LENGTH_LSAVE; i++){
		lsave[i] = (fortranLogical)0;
	}

	for (i = 0; i<LENGTH_ISAVE; i++){
		isave[i] = (int)0;
	}

	for (i = 0; i<LENGTH_DSAVE; i++){
		dsave[i] = 0;
	}

	/*---------------------------------------------------------------*/
	/* Group information in the column-wise */
	NposG = 3;

	Gcol_idx = NULL;
	Gcol_space = NULL;

	mptr_Mat1 = mxGetPr(prhs[NposG]);
	ir = mxGetIr(prhs[NposG]);
	jc = mxGetJc(prhs[NposG]);

	elements = (int)mxGetNzmax(prhs[NposG]);

	Gcol_idx = Malloc(struct feature_node*, mn_K);
	Gcol_space = Malloc(struct feature_node, elements + mn_K);

	j = 0;
	for (i = 0; i<mn_K; i++)
	{
		Gcol_idx[i] = &Gcol_space[j];

		low = (int)jc[i]; high = (int)jc[i + 1];

		for (k = low; k<high; k++)
		{
			Gcol_space[j].index = (int)ir[k];
			Gcol_space[j].value = mptr_Mat1[k];
			j++;
		}
		Gcol_space[j++].index = -1;
	}

	/* Group information in the row-wise */
	Grow_idx = NULL;
	Grow_space = NULL;

	Grhs[0] = mxDuplicateArray(prhs[NposG]);
	if (mexCallMATLAB(1, Glhs, 1, Grhs, "transpose")){
		mexPrintf("Error: cannot transpose training instance matrix\n");
	}

	mptr_Mat1 = mxGetPr(Glhs[0]);
	ir = mxGetIr(Glhs[0]);
	jc = mxGetJc(Glhs[0]);

	Grow_idx = Malloc(struct feature_node*, mn_Ksub);
	Grow_space = Malloc(struct feature_node, elements + mn_Ksub);

	j = 0;
	for (i = 0; i<mn_Ksub; i++)
	{
		Grow_idx[i] = &Grow_space[j];

		low = (int)jc[i]; high = (int)jc[i + 1];

		for (k = low; k<high; k++)
		{
			Grow_space[j].index = (int)ir[k];
			Grow_space[j].value = mptr_Mat1[k];
			j++;
		}
		Grow_space[j++].index = -1;
	}

	/* Laplacian matrix */
	NposG = 7;

	mptr_Mat1 = mxGetPr(prhs[NposG]);
	ir = mxGetIr(prhs[NposG]);
	jc = mxGetJc(prhs[NposG]);

	elements = (int)mxGetNzmax(prhs[NposG]);

	L_idx = Malloc(struct feature_node*, mn_J);
	L_space = Malloc(struct feature_node, elements + mn_J);

	j = 0;
	for (i = 0; i<mn_J; i++)
	{
		L_idx[i] = &L_space[j];

		low = (int)jc[i]; high = (int)jc[i + 1];

		for (k = low; k<high; k++)
		{
			L_space[j].index = (int)ir[k];
			L_space[j].value = mptr_Mat1[k];
			j++;
		}
		L_space[j++].index = -1;
	}

	/* SBtr */
	NposG = 9;

	mptr_Mat1 = mxGetPr(prhs[NposG]);
	ir = mxGetIr(prhs[NposG]);
	jc = mxGetJc(prhs[NposG]);

	elements = (int)mxGetNzmax(prhs[NposG]);

	SBtr_idx = Malloc(struct feature_node*, mn_K2);
	SBtr_space = Malloc(struct feature_node, elements + mn_K2);

	j = 0;
	for (i = 0; i<mn_K2; i++)
	{
		SBtr_idx[i] = &SBtr_space[j];

		low = (int)jc[i]; high = (int)jc[i + 1];

		for (k = low; k<high; k++)
		{
			SBtr_space[j].index = (int)ir[k];
			SBtr_space[j].value = mptr_Mat1[k];
			j++;
		}
		SBtr_space[j++].index = -1;
	}	

	/* SBdb */
	NposG = 11;

	mptr_Mat1 = mxGetPr(prhs[NposG]);
	ir = mxGetIr(prhs[NposG]);
	jc = mxGetJc(prhs[NposG]);

	elements = (int)mxGetNzmax(prhs[NposG]);

	SBdb_idx = Malloc(struct feature_node*, mn_J);
	SBdb_space = Malloc(struct feature_node, elements + mn_J);

	j = 0;
	for (i = 0; i<mn_J; i++)
	{
		SBdb_idx[i] = &SBdb_space[j];

		low = (int)jc[i]; high = (int)jc[i + 1];

		for (k = low; k<high; k++)
		{
			SBdb_space[j].index = (int)ir[k];
			SBdb_space[j].value = mptr_Mat1[k];
			j++;
		}
		SBdb_space[j++].index = -1;
	}
	/*---------------------------------------------------------------*/

	/* initial solution */
	f = 0.0;
	g = (double *)calloc(len_wdb, sizeof(double));
    memcpy(Wrow, mxGetPr(Wrhs[0]), len_wdb*sizeof(double));    
            
	outer_count = 0;
	for (iter = 0; iter<maxTotalIts; iter++){
		/* */
		setulb(&len_wdb, &m_Hess, Wrow, l, u, nbd, &f, g, &factr, &pgtol, wa, iwa, task, &iprint,
			csave, lsave, isave, dsave);
        
		if (strstr(task, "FG") != NULL){                         
			f = 0.0;
			memcpy(g, m_mGzeros, len_wdb*sizeof(double));
            
			/* transpose */
			memcpy(mxGetPr(Wrhs[0]), Wrow, len_wdb*sizeof(double));
			if (mexCallMATLAB(1, Wlhs, 1, Wrhs, "transpose")){
				mexPrintf("Error: cannot transpose training instance matrix\n");
			}
			Wcol = mxGetPr(Wlhs[0]);

            
            
			/* Wds = W.^2 */
			for (i = 0; i<len_wdb; i++){
				Wds[i] = Wrow[i] * Wrow[i];
			}

			/* Nestrov smooth approximation */
			mr_obj1 = 0.0;
			mr_obj2 = 0.0;
			mn_Cnt1 = 0;

			for (i = 0; i<mn_J; i++){
				mptr_Mat1 = (mm_Weight + i*mn_Ksub);
				mptr_Mat2 = (Wds + i*mn_K);

				for (j = 0; j<mn_Ksub; j++){
					mr_weight = rinner[j];

					mr_sum1 = 0.0;
					xi = Grow_idx[j];
					while (xi->index != -1){
						/* mr_sum1 += (xi->value) * mptr_Mat2[xi->index]; */
						mr_sum1 += mptr_Mat2[xi->index];
						xi++;
					}
					/* || Wg ||_2 */
					mr_sum2 = sqrt(mr_sum1);

					if (mr_sum2 > (mu / mr_weight)){
						mn_Cnt1++;

						mr_obj1 += (mr_weight*mr_sum2);
						mptr_Mat1[j] = mr_weight / mr_sum2;
					}
					else {
						mr_tmp1 = mr_weight*mr_weight;

						mr_obj2 += mr_tmp1*mr_sum1;
						mptr_Mat1[j] = mr_tmp1 / mu;
					}
				}
			}
			f = mr_obj1 - 0.5*(mu*mn_Cnt1) + 0.5*(mr_obj2 / mu);

			/* W - W0 at nonzero elements, row-wise */
			mr_sum1 = 0.0;
			for (i = 0; i<len_NZ; i++){
				idx1 = NzIDX[i];

				mr_tmp1 = Wrow[idx1] - W0[i];
				mr_sum1 += (mr_tmp1*mr_tmp1);

				g[idx1] = mr_tmp1;
			}
			f += 0.5*mr_sum1;

			/* Gradient update */
			mr_obj1 = 0.0;
			for (i = 0; i<mn_J; i++){
				idx1 = i*mn_K;

				mptr_Mat1 = g + idx1;
				mptr_Mat2 = SBZ + idx1;
				mptr_Mat3 = Wrow + idx1;

				mptr_Mat4 = mm_Weight + (i*mn_Ksub);

				xi_Ref1 = L_idx[i];
				xi_Ref2 = SBdb_idx[i];
				for (j = 0; j<mn_K; j++)
				{
					mr_tmp1 = mptr_Mat3[j];

					/* LW */
					xi = xi_Ref1;
					mptr_Mat5 = Wcol + (j*mn_J);

					mr_sum1 = 0.0;
					while (xi->index != -1){
						mr_sum1 += (xi->value) * mptr_Mat5[xi->index];
						xi++;
					}
					mr_obj2 = mr_sum1;
					mr_obj1 += (mr_tmp1*mr_sum1);

					/* Weight*G */
					xi = Gcol_idx[j];

					mr_sum1 = 0.0;
					while (xi->index != -1){
						/* mr_sum1 += (xi->value)*mptr_Mat4[xi->index]; */
						mr_sum1 += mptr_Mat4[xi->index];
						xi++;
					}
					mr_obj2 += (mr_tmp1*mr_sum1);

					/* SBdb*W */
					mptr_Mat5 = Wcol + (j*mn_J);
					xi = xi_Ref2;

					mr_sum1 = 0.0;
					while (xi->index != -1){
						mr_sum1 += (xi->value) * mptr_Mat5[xi->index];
						xi++;
					}
					mr_obj2 += mr_sum1;

					/* g */
					mptr_Mat1[j] += (mr_obj2 - mptr_Mat2[j]);
				}
			}
			f += 0.5*mr_obj1;

			/* sum(sum( (Z - W'*SB').^2 )) */
			mr_obj1 = 0.0;
			for (i = 0; i<mn_K; i++){
				mptr_Mat1 = Z + (i*mn_K2);
				mptr_Mat2 = Wcol + (i*mn_J);

				for (j = 0; j<mn_K2; j++){
					xi = SBtr_idx[j];

					mr_sum1 = 0.0;
					while (xi->index != -1){
						mr_sum1 += (xi->value) * mptr_Mat2[xi->index];
						xi++;
					}

					mr_tmp1 = mptr_Mat1[j] - mr_sum1;
					mr_obj1 += (mr_tmp1*mr_tmp1);
				}
			}

			/* for L1 loss */
			mr_obj2 = 0.0;
			for (i = 0; i<mn_KL1; i++){
				mr_tmp1 = rleafE[i];
				idx1 = L1IDX[i];

				mptr_Mat1 = Wcol + (idx1*mn_J);
				mr_sum1 = 0.0;
				for (j = 0; j<mn_J; j++){
					mr_sum1 += mptr_Mat1[j];
					g[j*mn_K + idx1] += mr_tmp1;
				}

				mr_obj2 += mr_tmp1*mr_sum1;
			}

			f += 0.5*(gamma*mr_obj1) + mr_obj2;
		}
		else if (strstr(task, "NEW_X") != NULL){
			objHist[outer_count] = (double)f;
			outer_count++;

			if (verbose>0){
				mexPrintf("Iteration %4d, f = %5.5e \n", outer_count, f);
			}

			if (outer_count >= maxIts){
				mexPrintf("Maxed-out iteration counter, exiting...");
				break;
			}
		}
		else{
			break;
		}
	}

	/* Outputs:
	* f, task, and the "save" variables: csave, lsave, isave, dsave
	*/
    plhs[0] = mxCreateDoubleMatrix(mn_K, mn_J, mxREAL);
	memcpy(mxGetPr(plhs[0]), Wrow, len_wdb*sizeof(double));
    
	if (outer_count == 0){
		objHist[0] = 0;	
		outer_count = 1;
	}	            
	plhs[1] = mxCreateDoubleMatrix(outer_count, 1, mxREAL);
	memcpy(mxGetPr(plhs[1]), objHist, outer_count*sizeof(double));

	free(g);
    
	free(Grow_idx);
	free(Grow_space);

	free(Gcol_idx);
	free(Gcol_space);
    
	free(L_idx);
	free(L_space);

    free(SBtr_idx);
	free(SBtr_space);

	free(SBdb_idx);
	free(SBdb_space); 

	mxFree(nbd);
	free(u);
	free(l);
    
    free(Wrow);
    free(Wds);
	free(m_mGzeros);
	free(mm_Weight);
    
	mxFree(wa);
	mxFree(iwa);

	return;
}
